import copy

import torch
import torch.nn as nn
import einops

from .backbone.point_net_pp import PointNetPP

# 这段代码定义了三个用于对象编码的神经网络模块：GTObjEncoder、PcdObjEncoder和ObjColorEncoder。
# 这些模块分别用于处理不同的对象特征，并将它们转换为嵌入向量。
#
# GTObjEncoder:
# 这个模块用于处理对象的类别或特征向量。
# 如果配置参数onehot_ft为True，则使用嵌入层将类别转换为嵌入向量。
# 否则，使用全连接层将特征向量转换为嵌入向量。
# 最后，应用层规范化和dropout。
# PcdObjEncoder:
#
# 这个模块用于处理点云数据。
# 它使用PointNet++架构来提取点云的局部和全局特征。
# 由于PointNet++的实现方式，代码中注释掉的部分是为了减少GPU内存的使用。
# 对每个对象分别处理点云数据，然后将结果堆叠起来。
# ObjColorEncoder:
#
# 这个模块用于处理对象的颜色信息。
# 它将颜色信息视为高斯混合模型（GMM）的权重和均值。
# 通过全连接层处理颜色均值，并与权重相乘，然后求和得到嵌入向量。
# 这些模块可以单独或组合使用，以处理不同类型的数据，并在多模态任务中提供对象的嵌入表示。例如，在3D物体检测或3D场景理解任务中，可以同时使用点云编码器和颜色编码器来获取更丰富的对象特征。


class GTObjEncoder(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()  # 调用父类构造函数进行初始化
        self.config = copy.deepcopy(config)  # 深拷贝配置，确保独立性
        self.config.hidden_size = hidden_size  # 设置隐藏层大小

        # 根据配置确定使用哪种类型的线性层
        if self.config.onehot_ft:
            # 如果使用one-hot特征，使用Embedding层将类别标签转换为嵌入向量
            self.ft_linear = [nn.Embedding(self.config.num_obj_classes, self.config.hidden_size)]
        else:
            # 否则，使用Linear层将特征向量转换为嵌入向量
            self.ft_linear = [nn.Linear(self.config.dim_ft, self.config.hidden_size)]

        # 将LayerNorm层添加到线性层列表中以进行层归一化
        self.ft_linear.append(nn.LayerNorm(self.config.hidden_size))
        # 使用nn.Sequential将层列表转换为一个顺序执行的模块
        self.ft_linear = nn.Sequential(*self.ft_linear)
        # 初始化dropout层
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, obj_fts):
        '''
        Args:
            obj_fts: LongTensor (batch, num_objs), 或者 FloatTensor (batch, num_objs, dim_ft)
        '''
        # 通过线性层（可能包括Embedding层）处理输入的对象特征
        obj_embeds = self.ft_linear(obj_fts)
        # 应用dropout
        obj_embeds = self.dropout(obj_embeds)
        # 返回处理后的对象嵌入向量
        return obj_embeds


class PcdObjEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()  # 调用父类构造函数进行初始化
        self.config = config  # 保存配置信息
        # 初始化PointNet++网络，配置包括点采样、半径和多层感知机配置
        self.pcd_net = PointNetPP(
            sa_n_points=config.sa_n_points,
            sa_n_samples=config.sa_n_samples,
            sa_radii=config.sa_radii,
            sa_mlps=config.sa_mlps,
        )
        # 初始化dropout层
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, obj_pcds):
        batch_size, num_objs, _, _ = obj_pcds.size()  # 获取输入点云的维度信息
        obj_embeds = []  # 初始化列表以保存每个对象的嵌入
        for i in range(batch_size):  # 遍历批次中的每个样本
            obj_embeds.append(self.pcd_net(obj_pcds[i]))  # 对每个样本的点云应用PointNet++，生成嵌入
        obj_embeds = torch.stack(obj_embeds, 0)  # 将列表中的嵌入堆叠成一个新的张量
        obj_embeds = self.dropout(obj_embeds)  # 应用dropout
        return obj_embeds  # 返回处理后的对象嵌入


class ObjColorEncoder(nn.Module):
    def __init__(self, hidden_size, dropout=0):
        super().__init__()  # 调用父类构造函数进行初始化
        # 定义一个序列化网络，用于将颜色信息编码为嵌入向量
        self.ft_linear = nn.Sequential(
            nn.Linear(3, hidden_size),  # 将3维颜色向量转换为隐藏层大小的向量
            nn.ReLU(),  # 使用ReLU激活函数增加非线性
            nn.LayerNorm(hidden_size, eps=1e-12),  # 使用层归一化稳定训练
            nn.Dropout(dropout)  # 应用dropout防止过拟合
        )

    def forward(self, obj_colors):
        # obj_colors: (batch, nobjs, 3, 4)
        gmm_weights = obj_colors[..., :1]  # 获取颜色混合模型的权重
        gmm_means = obj_colors[..., 1:]  # 获取颜色混合模型的均值
        # 对颜色均值进行编码并与权重相乘，然后沿特定维度求和以获得最终嵌入
        embeds = torch.sum(self.ft_linear(gmm_means) * gmm_weights, 2)
        return embeds  # 返回颜色的嵌入表示

        