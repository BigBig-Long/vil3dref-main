import os                  # 导入os库，用于操作操作系统功能
from pathlib import Path  # 导入Path类，用于路径操作
from pprint import pformat # 导入pformat，用于美化打印Python数据结构
import pickle             # 导入pickle库，用于对象的序列化和反序列化
import torch              # 导入PyTorch库，用于深度学习和张量操作
import torch.distributed as dist # 导入PyTorch的分布式模块

def load_init_param(opts):
    """
    Load parameters for the rendezvous distributed procedure
    """
    print(opts)  # 打印输入的参数
    # sync file 创建或检查同步文件的路径
    if opts.output_dir != "":
        sync_dir = Path(opts.output_dir).resolve()  # 确保路径是绝对路径
        sync_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
        sync_file = f"{sync_dir}/.torch_distributed_sync"  # 创建同步文件
    else:
        raise RuntimeError("Can't find any sync dir")  # 如果没有输出目录则报错

    # world size 处理全局设备数
    if opts.world_size != -1:
        world_size = opts.world_size
    elif os.environ.get("WORLD_SIZE", "") != "":
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        raise RuntimeError("Can't find any world size")  # 如果无法确定设备数则报错

    # rank 处理当前进程的全局排名
    if os.environ.get("RANK", "") != "":
        rank = int(os.environ["RANK"])
    else:
        if opts.node_rank != -1:
            node_rank = opts.node_rank
        elif os.environ.get("NODE_RANK", "") != "":
            node_rank = int(os.environ["NODE_RANK"])
        else:
            raise RuntimeError("Can't find any rank or node rank")
        if opts.local_rank != -1:
            local_rank = opts.local_rank
        elif os.environ.get("LOCAL_RANK", "") != "":
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            raise RuntimeError("Can't find any rank or local rank")
        n_gpus = torch.cuda.device_count()  # 获取当前节点的GPU数量
        rank = local_rank + node_rank * n_gpus  # 计算全局rank
    opts.rank = rank  # 设置opts的rank属性

    return {
        "backend": "nccl",  # 使用NCCL后端进行GPU之间的通信
        "init_method": f"file://{sync_file}",  # 使用文件初始化方法
        "rank": rank,
        "world_size": world_size,
    }

def init_distributed(opts):
    init_param = load_init_param(opts)  # 加载初始化参数
    rank = init_param["rank"]
    print(f"Init distributed {init_param['rank']} - {init_param['world_size']}")  # 打印初始化信息
    dist.init_process_group(**init_param)  # 初始化分布式进程组

def is_default_gpu(opts) -> bool:
    return opts.local_rank == -1 or dist.get_rank() == 0  # 判断当前GPU是否为默认GPU

def is_dist_avail_and_initialized():
    if not dist.is_available():  # 检查分布式是否可用
        return False
    if not dist.is_initialized():  # 检查分布式是否已经初始化
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():  # 检查分布式环境是否正常
        return 1
    return dist.get_world_size()  # 获取全局设备数

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()  # 获取全局设备数
    if world_size == 1:
        return [data]  # 如果只有一个设备，直接返回数据
    # serialized to a Tensor 序列化数据到Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")
    # obtain Tensor size of each rank 获取每个设备的Tensor大小
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    # receiving Tensor from all ranks 从所有设备接收Tensor
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()  # 获取全局设备数
    if world_size < 2:
        return input_dict  # 如果设备数少于2，直接返回输入字典
    with torch.no_grad():  # 禁用梯度计算
        names = []
        values = []
        # sort the keys so that they are consistent across processes 对键进行排序以保持一致性
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)  # 执行全局reduce操作
        if average:
            values /= world_size  # 如果需要平均，除以设备数
        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict
