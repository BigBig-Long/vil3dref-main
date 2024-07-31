# 从torch库中导入nn模块和Tensor类型
from torch import nn, Tensor

# 从外部工具库中导入PointnetSAModule模块
from ..external_tools.pointnet2.pointnet2_modules import PointnetSAModule

# 定义一个函数，用于将点云数据分割为位置和特征张量
def break_up_pc(pc: Tensor) -> [Tensor, Tensor]:
    """
    将点云分割为xyz位置和特征张量。
    此方法来自VoteNet代码库（https://github.com/facebookresearch/votenet）
    @param pc: 点云 [N, 3 + C]
    @return: xyz张量和特征张量
    """
    # 提取点云的xyz位置信息
    xyz = pc[..., 0:3].contiguous()
    # 如果点云有额外的特征，则提取特征，否则设置为None
    features = (
        pc[..., 3:].transpose(1, 2).contiguous()
        if pc.size(-1) > 3 else None
    )
    # 返回位置张量和特征张量
    return xyz, features

# 定义PointNet++编码器类
class PointNetPP(nn.Module):
    """
    Pointnet++编码器。
    对于超参数，请参阅论文（https://arxiv.org/abs/1706.02413）
    """
    def __init__(self, sa_n_points: list,
                 sa_n_samples: list,
                 sa_radii: list,
                 sa_mlps: list,
                 bn=True,
                 use_xyz=True):
        # 调用父类的初始化方法
        super().__init__()
        # 计算SA模块的数量
        n_sa = len(sa_n_points)
        # 检查超参数列表的长度是否一致
        if not (n_sa == len(sa_n_samples) == len(sa_radii) == len(sa_mlps)):
            raise ValueError('给定超参数的长度不兼容')
        # 初始化编码器模块列表
        self.encoder = nn.ModuleList()
        # 遍历每个SA模块
        for i in range(n_sa):
            # 添加PointnetSAModule到编码器列表中
            self.encoder.append(PointnetSAModule(
                npoint=sa_n_points[i],
                nsample=sa_n_samples[i],
                radius=sa_radii[i],
                mlp=sa_mlps[i],
                bn=bn,
                use_xyz=use_xyz,
            ))
        # 计算输出点的数量
        out_n_points = sa_n_points[-1] if sa_n_points[-1] is not None else 1
        # 初始化全连接层
        self.fc = nn.Linear(out_n_points * sa_mlps[-1][-1], sa_mlps[-1][-1])

    def forward(self, features):
        """
        @param features: B x N_objects x N_Points x 3 + C
        """
        # 分割点云数据为位置和特征
        xyz, features = break_up_pc(features)
        # 遍历每个编码器模块
        for i in range(len(self.encoder)):
            # 通过编码器模块处理点云数据
            xyz, features = self.encoder[i](xyz, features)
        # 通过全连接层处理特征
        return self.fc(features.view(features.size(0), -1))
