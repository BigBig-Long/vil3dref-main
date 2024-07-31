import random  # 导入random库，用于生成随机数
import numpy as np  # 导入NumPy库，用于进行高效的数值计算
from typing import Tuple, Union, Dict, Any  # 导入类型注解库，用于添加类型提示
import torch  # 导入PyTorch库，用于深度学习和张量运算
import torch.distributed as dist  # 导入PyTorch的分布式模块
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入PyTorch的分布式数据并行模块
from .distributed import init_distributed  # 导入初始化分布式设置的函数
from .logger import LOGGER  # 导入日志记录器

def set_random_seed(seed):
    random.seed(seed)  # 设定Python内置随机库的随机种子
    np.random.seed(seed)  # 设定NumPy随机库的随机种子
    torch.manual_seed(seed)  # 设定PyTorch随机库的随机种子
    torch.cuda.manual_seed_all(seed)  # 设定PyTorch在所有CUDA设备上的随机种子

def set_cuda(opts) -> Tuple[bool, int, torch.device]:
    """
    Initialize CUDA for distributed computing
    """
    if not torch.cuda.is_available():  # 检查CUDA是否可用
        assert opts.local_rank == -1, opts.local_rank  # 如果CUDA不可用，确保本地rank为-1
        return True, 0, torch.device("cpu")  # 返回使用CPU的设置

    # get device settings
    if opts.local_rank != -1:  # 如果指定了本地rank
        init_distributed(opts)  # 初始化分布式环境
        torch.cuda.set_device(opts.local_rank)  # 设定当前CUDA设备
        device = torch.device("cuda", opts.local_rank)  # 创建代表特定CUDA设备的设备对象
        n_gpu = 1  # 使用单个GPU
        default_gpu = dist.get_rank() == 0  # 检查当前设备是否为默认（主）GPU
        if default_gpu:
            LOGGER.info(f"Found {dist.get_world_size()} GPUs")  # 如果是默认GPU，记录GPU总数
    else:
        default_gpu = True
        device = torch.device("cuda")  # 使用默认CUDA设备
        n_gpu = torch.cuda.device_count()  # 获取可用的CUDA设备数
    return default_gpu, n_gpu, device

def wrap_model(
    model: torch.nn.Module, device: torch.device, local_rank: int
) -> torch.nn.Module:
    model.to(device)  # 将模型移动到指定的设备
    if local_rank != -1:  # 如果使用分布式并行
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)  # 将模型包装为分布式数据并行模型
    # At the time of DDP wrapping, parameters and buffers (i.e., model.state_dict())
    # on rank0 are broadcasted to all other ranks.
    elif torch.cuda.device_count() > 1:  # 如果有多个CUDA设备可用
        LOGGER.info("Using data parallel")  # 记录使用数据并行
        model = torch.nn.DataParallel(model)  # 将模型包装为数据并行模型
    return model

class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop  # 返回noop方法，用于未定义属性的访问

    def noop(self, *args, **kwargs):
        return  # 一个无操作方法，用于默认行为
