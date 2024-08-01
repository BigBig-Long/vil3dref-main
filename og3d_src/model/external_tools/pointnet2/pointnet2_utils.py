# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch
# 导入未来模块以保证代码的兼容性
from __future__ import (
    division,  # 除法，确保整数除法产生浮点结果
    absolute_import,  # 绝对导入，防止相对导入
    with_statement,  # 支持 with 语句
    print_function,  # 打印功能，确保使用新版的 print 函数
    unicode_literals,  # 字符串字面量是 Unicode 字符串
)

import torch  # 导入 PyTorch
from torch.autograd import Function  # 用于自定义 autograd 函数的基类
import torch.nn as nn  # 导入神经网络模块
import pytorch_utils as pt_utils  # 导入自定义的 PyTorch 工具模块

import sys  # 导入系统模块

# 尝试导入 builtins 模块，以访问 Python 的内置命名空间
try:
    import builtins
except:
    import builtins as builtins  # 如果导入失败，使用别名重新导入

# 尝试导入 PointNet2 的扩展模块 _ext
try:
    import pointnet2._ext as _ext
except ImportError:
    # 如果导入失败，检查是否在 PointNet2 的安装过程中，如果不是，则引发错误
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

# 如果需要，可以使用类型提示（在不依赖 `typing` 模块的情况下）
if False:
    from typing import *

class RandomDropout(nn.Module):
    # 初始化函数，设置dropout概率和是否就地操作
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()  # 调用父类的初始化方法
        self.p = p  # dropout概率
        self.inplace = inplace  # 是否就地修改输入数据

    # 前向传播函数
    def forward(self, X):
        # 生成一个[0, p]范围内的随机数
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        # 调用 pt_utils 中的 feature_dropout_no_scaling 函数应用特征dropout
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # 类型说明: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        使用迭代最远点采样法选取一组具有最大最小距离的点
        参数:
        ----------
        xyz : torch.Tensor
            (B, N, 3)形状的张量，其中N > npoint
        npoint : int32
            采样集中的特征点数量
        返回:
        -------
        torch.Tensor
            (B, npoint)形状的张量，包含采样集
        """
        # 调用底层扩展模块 _ext 实现最远点采样
        return _ext.furthest_point_sampling(xyz, npoint)

    @staticmethod
    def backward(ctx, a=None):
        # 最远点采样的反向传播不涉及梯度计算，因此返回None
        return None, None



furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # 类型说明: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        参数:
        ----------
        features : torch.Tensor
            (B, C, N) 形状的张量，其中 B 是批次大小，C 是特征通道数，N 是点数
        idx : torch.Tensor
            (B, npoint) 形状的张量，表示要收集的特征的索引
        返回:
        -------
        torch.Tensor
            (B, C, npoint) 形状的张量
        """
        # 解析输入特征的尺寸
        _, C, N = features.size()
        # 保存反向传播需要用到的数据
        ctx.for_backwards = (idx, C, N)
        # 调用底层扩展模块 _ext 实现特征收集
        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # 从上下文中恢复反向传播需要用到的数据
        idx, C, N = ctx.for_backwards
        # 调用底层扩展模块 _ext 实现特征收集的梯度计算
        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        # 返回计算得到的梯度，对 idx 的梯度为 None
        return grad_features, None



gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # 类型说明: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        在已知点集中找到未知点集的三个最近邻
        参数:
        ----------
        unknown : torch.Tensor
            (B, n, 3) 形状的张量，表示未知特征点集
        known : torch.Tensor
            (B, m, 3) 形状的张量，表示已知特征点集
        返回:
        -------
        dist : torch.Tensor
            (B, n, 3) 形状的张量，表示到三个最近邻的 L2 距离
        idx : torch.Tensor
            (B, n, 3) 形状的张量，表示三个最近邻的索引
        """
        # 调用底层扩展模块 _ext 实现三最近邻查找
        dist2, idx = _ext.three_nn(unknown, known)
        # 返回距离的平方根（即真实的 L2 距离）和索引
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        # 三最近邻查找的反向传播不涉及梯度计算，因此返回 None
        return None, None



three_nn = ThreeNN.apply

class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # 类型说明: (Any, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        在三个特征点上执行加权线性插值
        参数:
        ----------
        features : torch.Tensor
            (B, c, m) 需要从中插值的特征描述符
        idx : torch.Tensor
            (B, n, 3) 目标特征点在 features 中的三个最近邻
        weight : torch.Tensor
            (B, n, 3) 权重
        返回:
        -------
        torch.Tensor
            (B, c, n) 插值后的特征张量
        """
        B, c, m = features.size()
        n = idx.size(1)
        # 保存反向传播所需的信息
        ctx.three_interpolate_for_backward = (idx, weight, m)
        # 调用扩展模块 _ext 执行三点插值
        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # 类型说明: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        参数:
        ----------
        grad_out : torch.Tensor
            (B, c, n) 输出梯度张量
        返回:
        -------
        grad_features : torch.Tensor
            (B, c, m) 特征的梯度张量
        None
        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        # 调用扩展模块 _ext 计算插值的梯度
        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )
        return grad_features, None, None



three_interpolate = ThreeInterpolate.apply

class GroupingOperation(Function):  # 定义一个名为GroupingOperation的自定义PyTorch函数类
    @staticmethod
    def forward(ctx, features, idx):  # 前向传播方法，接收特征张量和索引张量作为输入
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        features : torch.Tensor  # (B, C, N) 特征张量，其中包含要分组的特征
        idx : torch.Tensor  # (B, npoint, nsample) 张量，包含用于分组特征的索引
        Returns
        -------
        torch.Tensor  # 返回一个张量 (B, C, npoint, nsample)，表示分组后的特征
        """
        B, nfeatures, nsample = idx.size()  # 获取索引张量的尺寸，分别代表批量大小、每个点的特征数和样本数
        _, C, N = features.size()  # 获取特征张量的尺寸，分别代表批量大小、通道数和总点数
        ctx.for_backwards = (idx, N)  # 保存索引张量和总点数，以便在反向传播时使用
        return _ext.group_points(features, idx)  # 调用自定义扩展函数执行分组操作

    @staticmethod
    def backward(ctx, grad_out):  # 反向传播方法，接收输出张量的梯度作为输入
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor  # (B, C, npoint, nsample) 输出张量在前的梯度
        Returns
        -------
        torch.Tensor  # 返回特征张量的梯度 (B, C, N)
        None  # 索引张量的梯度，因为索引不是可微分的，所以返回None
        """
        idx, N = ctx.for_backwards  # 从上下文中获取保存的索引张量和总点数
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)  # 调用自定义扩展函数计算特征张量的梯度
        return grad_features, None  # 返回特征张量的梯度和None



grouping_operation = GroupingOperation.apply
class BallQuery(Function):  # 定义一个名为BallQuery的自定义PyTorch函数类
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):  # 前向传播方法，接收球查询的参数和坐标张量
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        radius : float  # 球查询的半径
        nsample : int  # 每个球中特征的最大数量
        xyz : torch.Tensor  # (B, N, 3) 特征的xyz坐标
        new_xyz : torch.Tensor  # (B, npoint, 3) 球查询的中心坐标
        Returns
        -------
        torch.Tensor  # 返回一个张量 (B, npoint, nsample)，包含形成查询球的特征的索引
        """
        return _ext.ball_query(new_xyz, xyz, radius, nsample)  # 调用自定义扩展函数执行球查询

    @staticmethod
    def backward(ctx, a=None):  # 反向传播方法
        return None, None, None, None  # 因为球查询操作不需要在反向传播中计算梯度，所以返回四个None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    使用球查询来分组的模块
    Parameters
    ---------
    radius : float32
    球查询的半径
    nsample : int32
    在球中要聚集的最大特征数量
    """
    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()  # 调用父类的构造函数
        self.radius = radius  # 球查询的半径
        self.nsample = nsample  # 每个球中的最大特征数量
        self.use_xyz = use_xyz  # 是否使用xyz坐标作为特征
        self.ret_grouped_xyz = ret_grouped_xyz  # 是否返回分组后的xyz坐标
        self.normalize_xyz = normalize_xyz  # 是否对分组后的xyz坐标进行归一化
        self.sample_uniformly = sample_uniformly  # 是否均匀地采样点
        self.ret_unique_cnt = ret_unique_cnt  # 是否返回每个球中唯一点的数量

        # 如果需要返回每个球中唯一点的数量，则必须均匀采样
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
        特征的xyz坐标 (B, N, 3)
        new_xyz : torch.Tensor
        中心点坐标 (B, npoint, 3)
        features : torch.Tensor
        特征描述符 (B, C, N)
        Returns
        -------
        new_features : torch.Tensor
        返回一个张量 (B, 3 + C, npoint, nsample)，包含分组后的特征
        """
        # 执行球查询，获取每个中心点周围的点的索引
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        # 如果需要均匀采样，则对每个球内的点进行均匀采样
        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))  # 初始化唯一点计数
            # 遍历每个批次和每个区域（中心点）
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])  # 获取每个球内的唯一点索引
                    num_unique = unique_ind.shape[0]  # 计算唯一点的数量
                    unique_cnt[i_batch, i_region] = num_unique  # 更新计数
                    # 如果唯一点的数量少于nsample，则随机选择额外的点以补足
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))  # 合并索引
                    idx[i_batch, i_region, :] = all_ind  # 更新索引

        # 转置xyz坐标并执行分组操作
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        # 将分组后的坐标减去中心点坐标
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        # 如果需要归一化，则对分组后的坐标进行归一化
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        # 如果提供了特征描述符，则对这些特征进行分组
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            # 如果需要包含xyz坐标作为特征，则将它们与分组后的特征合并
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            # 如果没有提供特征描述符，则必须使用xyz坐标作为特征
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        # 准备返回值
        ret = [new_features]
        # 如果需要返回分组后的xyz坐标，则添加到返回列表
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        # 如果需要返回每个球中唯一点的数量，则添加到返回列表
        if self.ret_unique_cnt:
            ret.append(unique_cnt)

        # 返回单个张量或元组
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    r"""
    将所有特征进行分组
    Parameters
    ---------
    """
    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz  # 是否使用xyz坐标作为特征

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
        特征的xyz坐标 (B, N, 3)
        new_xyz : torch.Tensor
        被忽略的参数，因为这里不使用中心点进行分组
        features : torch.Tensor
        特征描述符 (B, C, N)
        Returns
        -------
        new_features : torch.Tensor
        返回一个张量 (B, C + 3, 1, N)，包含分组后的特征
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)  # 转置xyz坐标并增加一个维度，为后续拼接做准备

        # 如果提供了特征描述符，则增加一个维度以便与xyz坐标拼接
        if features is not None:
            grouped_features = features.unsqueeze(2)

        # 如果需要包含xyz坐标作为特征，则将它们与分组后的特征描述符合并
        if self.use_xyz:
            new_features = torch.cat(
                [grouped_xyz, grouped_features], dim=1
            )  # (B, 3 + C, 1, N)
        else:
            new_features = grouped_features if features is not None else grouped_xyz

        # 返回分组后的特征
        return new_features

