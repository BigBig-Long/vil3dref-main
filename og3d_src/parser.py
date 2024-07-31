import argparse  # 导入argparse库，用于命令行参数解析
import sys       # 导入sys库，用于访问与Python解释器紧密相关的变量和函数
import yaml      # 导入yaml库，用于处理YAML文件

def load_parser():
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
    parser.add_argument("--config", required=True, help="YAML config files")  # 添加一个必需的命令行参数--config
    parser.add_argument('--output_dir')  # 添加一个可选的命令行参数--output_dir
    parser.add_argument('--resume_files', nargs='+')  # 添加一个接受一个或多个值的命令行参数--resume_files
    parser.add_argument('--resume_optimizer', default=None)  # 添加一个有默认值的命令行参数--resume_optimizer
    parser.add_argument('--test', default=False, action='store_true')  # 添加一个布尔类型的命令行参数--test
    # 分布式计算相关的命令行参数
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="local rank for distributed training on gpus",  # 添加一个整型命令行参数--local_rank，用于GPU分布式训练的本地排名
    )
    parser.add_argument(
        "--node_rank", type=int, default=0,
        help="Id of the node",  # 添加一个整型命令行参数--node_rank，用于标识节点的ID
    )
    parser.add_argument(
        "--world_size", type=int, default=1,
        help="Number of GPUs across all nodes",  # 添加一个整型命令行参数--world_size，用于指定所有节点上GPU的数量
    )
    return parser  # 返回配置好的parser对象

def parse_with_config(parser):
    args = parser.parse_args()  # 解析命令行参数，并返回包含参数值的命名空间
    if args.config is not None:  # 检查是否提供了配置文件
        config_args = yaml.safe_load(open(args.config))  # 加载YAML配置文件内容
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }  # 获取命令行中指定的参数名，用于后续判断哪些参数可以被配置文件覆盖
        for k, v in config_args.items():  # 遍历配置文件中的参数
            if k not in override_keys:  # 如果配置文件中的参数没有在命令行中显式指定
                setattr(args, k, v)  # 则使用配置文件中的值覆盖args中的对应值
        del args.config  # 删除args中的config属性，因为后续不再需要
    return args  # 返回最终的args对象，包含了命令行和配置文件中指定的参数值
