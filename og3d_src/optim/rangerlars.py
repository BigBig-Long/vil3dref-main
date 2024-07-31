import torch, math  # 导入torch和math库
from torch.optim.optimizer import Optimizer  # 从torch.optim模块导入Optimizer类
import itertools as it  # 导入itertools库，通常用于创建迭代器
from .lookahead import *  # 从同一目录下的lookahead模块导入所有内容
from .ralamb import *  # 从同一目录下的ralamb模块导入所有内容

# 定义RangerLars函数，该函数接收优化器参数和Lookahead的参数
def RangerLars(params, alpha=0.5, k=6, *args, **kwargs):
    ralamb = Ralamb(params, *args, **kwargs)  # 创建一个Ralamb实例
    return Lookahead(ralamb, alpha, k)  # 将Ralamb实例封装进Lookahead策略，并返回
