# 导入必要的PyTorch库和自定义工具
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
# 获取当前文件的目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径，以便可以导入当前目录下的模块
sys.path.append(BASE_DIR)
# 导入点云处理相关的工具函数
import pointnet2_utils
# 导入PyTorch工具函数
import pytorch_utils as pt_utils
# 导入List类型
from typing import List


# 定义一个基础模块，用于点云的特征抽象
class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None  # 用于指定输出的点数
        self.groupers = None  # 用于分组操作的模块
        self.mlps = None  # 多层感知机模块

    # 前向传播函数
    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        # 初始化一个列表，用于存储新的特征
        new_features_list = []
        # 将输入的点云坐标进行转置，以便进行后续操作
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        # 如果指定了输出的点数，则进行最远点采样
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped,
            pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        ).transpose(1, 2).contiguous() if self.npoint is not None else None
        # 遍历所有的分组器
        for i in range(len(self.groupers)):
            # 对点云进行分组，并获取每个组内的特征
            new_features = self.groupers[i](xyz, new_xyz, features)
            # 通过多层感知机处理分组后的特征
            new_features = self.mlps[i](new_features)
            # 对特征进行最大池化，以获取每个组的主要特征
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            # 去掉最后一维，因为池化后只有一个值
            new_features = new_features.squeeze(-1)
            # 将处理后的特征添加到列表中
            new_features_list.append(new_features)
        # 将所有处理后的特征合并在一起
        return new_xyz, torch.cat(new_features_list, dim=1)


# 定义一个带有多尺度分组（MSG）的PointNet SA模块
class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet集抽象层，具有多尺度分组功能
    参数
    ----------
    npoint : int
        输出特征点的数量
    radii : list of float32
        用于分组的半径列表
    nsamples : list of int32
        每个球查询中的采样点数
    mlps : list of list of int32
        每个尺度下全局最大池化之前的PointNet规格
    bn : bool
        是否使用批量归一化
    use_xyz : bool
        是否将xyz坐标作为特征使用
    sample_uniformly : bool
        是否在局部区域内均匀采样
    """
    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True, use_xyz: bool = True, sample_uniformly: bool = False):
        super().__init__()
        # 确保输入的半径、采样数和MLP规格长度一致
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        # 初始化分组器和MLP模块列表
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        # 遍历每个尺度的配置
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            # 根据是否指定了输出点数，选择不同的分组器
            if npoint is not None:
                self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly))
            else:
                self.groupers.append(pointnet2_utils.GroupAll(use_xyz))
            # 准备MLP规格，如果使用xyz坐标，则增加输入维度
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            # 添加MLP模块
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

# 定义一个简化的PointNet SA模块，继承自PointnetSAModuleMSG
class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet集抽象层
    参数
    ----------
    mlp : list
        全局最大池化之前的PointNet规格
    npoint : int, optional
        输出特征点的数量
    radius : float, optional
        球的半径
    nsample : int, optional
        球查询中的采样点数
    bn : bool, optional
        是否使用批量归一化
    use_xyz : bool, optional
        是否将xyz坐标作为特征使用
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True):
        # 调用父类的构造函数，简化参数传递
        super().__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz
        )



# 定义一个带有投票功能的PointNet SA模块
class PointnetSAModuleVotes(nn.Module):
    r'''基于_PointnetSAModuleBase和PointnetSAModuleMSG修改，额外支持返回点索引以获取它们的GT投票
    参数
    ----------
    mlp : list
        MLP规格
    npoint : int, optional
        输出特征点的数量
    radius : float, optional
        球的半径
    nsample : int, optional
        球查询中的采样点数
    bn : bool, optional
        是否使用批量归一化
    use_xyz : bool, optional
        是否将xyz坐标作为特征使用
    pooling : str
        池化类型（'max', 'avg', 'rbf'）
    sigma : float, optional
        用于RBF池化的sigma值
    normalize_xyz : bool, optional
        是否将局部XYZ与半径归一化
    sample_uniformly : bool, optional
        是否在局部区域内均匀采样
    ret_unique_cnt : bool, optional
        是否返回唯一的计数
    '''
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True, pooling: str = 'max', sigma: float = None, normalize_xyz: bool = False, sample_uniformly: bool = False, ret_unique_cnt: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        # 根据是否指定了输出点数，选择不同的分组器
        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz, sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)
        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        r'''
        参数
        ----------
        xyz : torch.Tensor
            (B, N, 3) 张量，包含特征的xyz坐标
        features : torch.Tensor
            (B, C, N) 张量，包含特征的描述符
        inds : torch.Tensor
            (B, npoint) 张量，存储指向xyz点的索引（值在0-N-1之间）
        返回
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) 张量，包含新特征的xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) 张量，包含新特征的描述符
        inds: torch.Tensor
            (B, npoint) 张量，包含索引
        '''
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(xyz, new_xyz, features)
        new_features = self.mlp_module(grouped_features)
        if self.pooling == 'max':
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        elif self.pooling == 'avg':
            new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        elif self.pooling == 'rbf':
            rbf = torch.exp(-1 * grouped_xyz.pow(2).sum(1, keepdim=False) /(self.sigma**2) / 2)  # (B, npoint, nsample)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(self.nsample)  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt

# 继续注释：
# 如果没有返回唯一的计数，则只返回新的xyz坐标、新的特征和索引。
# 如果返回了唯一的计数，则还会返回unique_cnt，它表示每个局部区域中独特的点的数量。
# 这个模块的输出可以用于进一步的点云处理，例如点云分割或目标检测。


# 定义一个带有投票功能的多尺度分组（MSG）PointNet SA模块
class PointnetSAModuleMSGVotes(nn.Module):
    r'''基于_PointnetSAModuleBase和PointnetSAModuleMSG修改，额外支持返回点索引以获取它们的GT投票
    参数
    ----------
    mlps : list of list of int32
        多尺度MLP规格
    npoint : int
        输出特征点的数量
    radii : list of float32
        分组半径列表
    nsamples : list of int32
        每个球查询中的采样点数
    bn : bool, optional
        是否使用批量归一化
    use_xyz : bool, optional
        是否将xyz坐标作为特征使用
    sample_uniformly : bool, optional
        是否在局部区域内均匀采样
    '''
    def __init__(self, *, mlps: List[List[int]], npoint: int, radii: List[float], nsamples: List[int], bn: bool = True, use_xyz: bool = True, sample_uniformly: bool = False):
        super().__init__()
        # 确保MLP规格、采样数和半径的长度一致
        assert(len(mlps) == len(nsamples) == len(radii))
        self.npoint = npoint
        # 初始化分组器和MLP模块列表
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        # 遍历每个尺度的配置
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            # 根据是否指定了输出点数，选择不同的分组器
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        r'''
        参数
        ----------
        xyz : torch.Tensor
            (B, N, 3) 张量，包含特征的xyz坐标
        features : torch.Tensor
            (B, C, N) 张量，包含特征的描述符
        inds : torch.Tensor
            (B, npoint) 张量，存储指向xyz点的索引（值在0-N-1之间）
        返回
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) 张量，包含新特征的xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) 张量，包含新特征的描述符
        inds: torch.Tensor
            (B, npoint) 张量，包含索引
        '''
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
        # 遍历每个分组器和MLP模块
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        # 将多个尺度的特征拼接在一起
        new_features = torch.cat(new_features_list, dim=1)
        return new_xyz, new_features, inds


# 定义一个用于特征传播的PointNet FP模块
class PointnetFPModule(nn.Module):
    r'''将一组特征传播到另一组
    参数
    ----------
    mlp : list
        PointNet模块参数
    bn : bool
        是否使用批量归一化
    '''
    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        r'''
        参数
        ----------
        unknown : torch.Tensor
            (B, n, 3) 张量，包含未知特征的xyz位置
        known : torch.Tensor
            (B, m, 3) 张量，包含已知特征的xyz位置
        unknow_feats : torch.Tensor
            (B, C1, n) 张量，待传播的特征
        known_feats : torch.Tensor
            (B, C2, m) 张量，待传播的特征
        返回
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) 张量，未知特征的新特征
        '''
        if known is not None:
            # 计算未知特征和已知特征之间的距离和索引
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            # 计算距离的倒数
            dist_recip = 1.0 / (dist + 1e-8)
            # 计算归一化因子
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            # 计算权重
            weight = dist_recip / norm
            # 对已知特征进行插值
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            # 如果没有已知特征，则直接扩展未知特征
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            # 将插值特征和未知特征拼接
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats

        # 将特征unsqueeze并在最后添加一个维度
        new_features = new_features.unsqueeze(-1)
        # 通过MLP处理特征
        new_features = self.mlp(new_features)
        # 返回最终的特征，移除最后一维
        return new_features.squeeze(-1)


# 定义一个可学习的特征传播层，基于多尺度分组（MSG）
class PointnetLFPModuleMSG(nn.Module):
    r'''基于_PointnetSAModuleBase和PointnetSAModuleMSG修改，可学习的特征传播层。
    参数
    ----------
    mlps : list of list of int32
        多尺度MLP规格
    radii : list of float32
        分组半径列表
    nsamples : list of int32
        每个球查询中的采样点数
    post_mlp : list of int32
        后处理MLP规格
    bn : bool, optional
        是否使用批量归一化
    use_xyz : bool, optional
        是否将xyz坐标作为特征使用
    sample_uniformly : bool, optional
        是否在局部区域内均匀采样
    '''
    def __init__(self, *, mlps: List[List[int]], radii: List[float], nsamples: List[int], post_mlp: List[int], bn: bool = True, use_xyz: bool = True, sample_uniformly: bool = False):
        super().__init__()
        # 确保MLP规格、采样数和半径的长度一致
        assert(len(mlps) == len(nsamples) == len(radii))
        self.post_mlp = pt_utils.SharedMLP(post_mlp, bn=bn)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        # 遍历每个尺度的配置
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            # 根据是否指定了输出点数，选择不同的分组器
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

    def forward(self, xyz2: torch.Tensor, xyz1: torch.Tensor, features2: torch.Tensor, features1: torch.Tensor) -> torch.Tensor:
        r'''从xyz1传播特征到xyz2。
        参数
        ----------
        xyz2 : torch.Tensor
            (B, N2, 3) 张量，包含特征的xyz坐标
        xyz1 : torch.Tensor
            (B, N1, 3) 张量，包含特征的xyz坐标
        features2 : torch.Tensor
            (B, C2, N2) 张量，包含特征的描述符
        features1 : torch.Tensor
            (B, C1, N1) 张量，包含特征的描述符
        返回
        -------
        new_features1 : torch.Tensor
            (B, \sum_k(mlps[k][-1]), N1) 张量，新特征的描述符
        '''
        new_features_list = []
        # 遍历每个分组器和MLP模块
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz1, xyz2, features1)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            if features2 is not None:
                new_features = torch.cat([new_features, features2], dim=1)
            new_features = new_features.unsqueeze(-1)
            new_features = self.post_mlp(new_features)
            new_features_list.append(new_features)
        # 将多个尺度的特征拼接在一起并移除最后一维
        return torch.cat(new_features_list, dim=1).squeeze(-1)


# 当这个文件作为主程序运行时，执行以下代码
if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    import numpy as np

    # 设置随机种子以确保可重复性
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # 创建一些随机变量作为输入数据
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    # 创建一个PointnetSAModuleMSG实例
    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()  # 将模块移动到GPU

    # 测试模块的前向传播
    print(test_module(xyz, xyz_feats))

    # 进行反向传播
    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(
            torch.cuda.FloatTensor(*new_features.size()).fill_(1)
        )

    # 打印新的特征和xyz的梯度
    print(new_features)
    print(xyz.grad)

