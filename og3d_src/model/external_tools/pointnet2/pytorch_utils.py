# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on Ref: https://github.com/erikwijmans/Pointnet2_PyTorch '''
import torch
import torch.nn as nn
from typing import List, Tuple

class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args: List[int],  # 一个整数列表，指定每一层的输出通道数
        *,
        bn: bool = False,  # 是否在每个卷积层后添加批量归一化（Batch Normalization）
        activation=nn.ReLU(inplace=True),  # 激活函数，默认为ReLU，支持原地操作（inplace=True）
        preact: bool = False,  # 是否在卷积之前应用激活函数和批量归一化
        first: bool = False,  # 是否为网络的第一层，影响是否添加批量归一化
        name: str = ""  # 用于命名层的字符串，有助于区分不同的SharedMLP实例
    ):
        super().__init__()
        # 遍历每一层（除了最后一层），创建Conv2d模块并添加到序列中
        for i in range(len(args) - 1):
            # 创建Conv2d模块，设置当前层的输入和输出通道数
            self.add_module(
                name + 'layer{}'.format(i),  # 使用name参数为每一层命名
                Conv2d(
                    args[i],  # 当前层的输入通道数
                    args[i + 1],  # 下一层的输出通道数
                    bn=(not first or not preact or (i != 0)) and bn,  # 根据条件决定是否添加批量归一化
                    activation=activation  # 根据条件决定是否添加激活函数
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact  # 是否在卷积之前应用激活函数和批量归一化
                )
            )


# _BNBase是一个基类，用于创建具有特定维度的批量归一化层
class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        # 添加一个批量归一化层，其类型由batch_norm参数指定
        self.add_module(name + "bn", batch_norm(in_size))
        # 初始化权重为1.0，偏置为0
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)

# BatchNorm1d是一个创建1D批量归一化层的类
class BatchNorm1d(_BNBase):
    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)

# BatchNorm2d是一个创建2D批量归一化层的类
class BatchNorm2d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)

# BatchNorm3d是一个创建3D批量归一化层的类
class BatchNorm3d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class _ConvBase(nn.Sequential):
    def __init__(
            self,
            in_size,  # 输入通道数
            out_size,  # 输出通道数
            kernel_size,  # 卷积核大小
            stride,  # 步长
            padding,  # 填充
            activation,  # 激活函数
            bn,  # 是否使用批量归一化
            init,  # 初始化函数，用于初始化卷积核权重
            conv=None,  # 卷积类，默认为nn.Conv2d
            batch_norm=None,  # 批量归一化类，默认为nn.BatchNorm2d
            bias=True,  # 是否在卷积层中使用偏置
            preact=False,  # 是否在卷积之前应用批量归一化和激活函数
            name=""  # 模块名称，有助于区分不同的_ConvBase实例
    ):
        super().__init__()
        # 如果使用批量归一化，则不使用卷积层的偏置
        bias = bias and (not bn)

        # 创建卷积单元
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        # 初始化卷积核权重
        init(conv_unit.weight)
        # 如果卷积层有偏置，则初始化偏置为0
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        # 如果使用批量归一化
        if bn:
            # 如果在卷积之前应用批量归一化
            if preact:
                bn_unit = batch_norm(in_size)
            else:
                bn_unit = batch_norm(out_size)

        # 如果在卷积之前应用批量归一化和激活函数
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            self.add_module(name + 'conv', conv_unit)
        # 如果在卷积之后应用批量归一化和激活函数
        if not preact:
            self.add_module(name + 'conv', conv_unit)
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)

class Conv1d(_ConvBase):
    def __init__(
        self,
        in_size: int,     # 输入通道数
        out_size: int,    # 输出通道数
        *,
        kernel_size: int = 1,    # 卷积核大小，默认为1
        stride: int = 1,         # 步长，默认为1
        padding: int = 0,        # 填充，默认为0
        activation=nn.ReLU(inplace=True),  # 激活函数，默认为ReLU，并且使用inplace=True
        bn: bool = False,         # 是否使用批量归一化，默认为False
        init=nn.init.kaiming_normal_,  # 权重初始化方法，默认为Kaiming正态分布初始化
        bias: bool = True,        # 是否在卷积层中使用偏置，默认为True
        preact: bool = False,     # 是否在卷积之前应用批量归一化和激活函数，默认为False
        name: str = ""            # 模块名称，默认为空字符串
    ):
        # 调用基类_ConvBase的构造函数，指定1D卷积和批量归一化的类
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,        # 使用nn.Conv1d进行1D卷积
            batch_norm=BatchNorm1d,# 使用BatchNorm1d进行批量归一化
            bias=bias,
            preact=preact,
            name=name
        )

class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size: int,         # 输入通道数
        out_size: int,        # 输出通道数
        *,
        kernel_size: Tuple[int, int] = (1, 1),  # 卷积核大小，默认为(1, 1)
        stride: Tuple[int, int] = (1, 1),       # 步长，默认为(1, 1)
        padding: Tuple[int, int] = (0, 0),      # 填充，默认为(0, 0)
        activation=nn.ReLU(inplace=True),       # 激活函数，默认为ReLU，并且使用inplace=True
        bn: bool = False,                         # 是否使用批量归一化，默认为False
        init=nn.init.kaiming_normal_,            # 权重初始化方法，默认为Kaiming正态分布初始化
        bias: bool = True,                        # 是否在卷积层中使用偏置，默认为True
        preact: bool = False,                     # 是否在卷积之前应用批量归一化和激活函数，默认为False
        name: str = ""                            # 模块名称，默认为空字符串
    ):
        # 调用基类_ConvBase的构造函数，指定2D卷积和批量归一化的类
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,        # 使用nn.Conv2d进行2D卷积
            batch_norm=BatchNorm2d,# 使用BatchNorm2d进行批量归一化
            bias=bias,
            preact=preact,
            name=name
        )


class Conv3d(_ConvBase):
    def __init__(
        self,
        in_size: int,           # 输入通道数
        out_size: int,          # 输出通道数
        *,
        kernel_size: Tuple[int, int, int] = (1, 1, 1),  # 卷积核大小，默认为(1, 1, 1)
        stride: Tuple[int, int, int] = (1, 1, 1),       # 步长，默认为(1, 1, 1)
        padding: Tuple[int, int, int] = (0, 0, 0),      # 填充，默认为(0, 0, 0)
        activation=nn.ReLU(inplace=True),              # 激活函数，默认为ReLU，并且使用inplace=True
        bn: bool = False,                               # 是否使用批量归一化，默认为False
        init=nn.init.kaiming_normal_,                   # 权重初始化方法，默认为Kaiming正态分布初始化
        bias: bool = True,                              # 是否在卷积层中使用偏置，默认为True
        preact: bool = False,                           # 是否在卷积之前应用批量归一化和激活函数，默认为False
        name: str = ""                                  # 模块名称，默认为空字符串
    ):
        # 调用基类_ConvBase的构造函数，指定3D卷积和批量归一化的类
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv3d,        # 使用nn.Conv3d进行3D卷积
            batch_norm=BatchNorm3d,# 使用BatchNorm3d进行批量归一化
            bias=bias,
            preact=preact,
            name=name
        )


class FC(nn.Sequential):
    def __init__(
        self,
        in_size: int,       # 输入特征数量
        out_size: int,      # 输出特征数量
        *,
        activation=nn.ReLU(inplace=True),  # 激活函数，默认为ReLU，并且使用inplace=True
        bn: bool = False,                   # 是否使用批量归一化，默认为False
        init=None,                         # 权重初始化方法，默认为None
        preact: bool = False,               # 是否在全连接层之前应用批量归一化和激活函数，默认为False
        name: str = ""                      # 模块名称，默认为空字符串
    ):
        super().__init__()
        # 创建全连接层
        fc = nn.Linear(in_size, out_size, bias=not bn)
        # 如果提供了初始化方法，则初始化权重
        if init is not None:
            init(fc.weight)
        # 如果不使用批量归一化，则初始化偏置为0
        if not bn:
            nn.init.constant_(fc.bias, 0)
        # 如果使用preact，先添加批量归一化和激活函数
        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)
        # 添加全连接层
        self.add_module(name + 'fc', fc)
        # 如果不使用preact，则在全连接层之后添加批量归一化和激活函数
        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)
def set_bn_momentum_default(bn_momentum):
    # 定义一个函数fn，用于设置Batch Normalization层的动量
    def fn(m):
        # 检查模块m是否是Batch Normalization层的实例
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # 如果是，则设置其动量为bn_momentum
            m.momentum = bn_momentum
    # 返回这个函数
    return fn

class BNMomentumScheduler(object):
    def __init__(
        self, model, bn_lambda, last_epoch=-1,
        setter=set_bn_momentum_default
    ):
        # 检查model是否是nn.Module的实例
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )
        # 保存模型和setter函数
        self.model = model
        self.setter = setter
        # 保存用于计算动量的lambda函数
        self.lmbd = bn_lambda
        # 初始化时调用step函数，设置初始动量
        self.step(last_epoch + 1)
        # 保存最后一个epoch的编号
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        # 如果没有提供epoch，则默认为上一个epoch加1
        if epoch is None:
            epoch = self.last_epoch + 1
        # 更新最后一个epoch的编号
        self.last_epoch = epoch
        # 使用setter函数和lambda函数计算出的动量更新模型中所有BN层的动量
        self.model.apply(self.setter(self.lmbd(epoch)))



