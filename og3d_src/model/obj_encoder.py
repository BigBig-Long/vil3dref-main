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
        super().__init__()
        self.config = copy.deepcopy(config)
        self.config.hidden_size = hidden_size

        if self.config.onehot_ft:
            self.ft_linear = [nn.Embedding(self.config.num_obj_classes, self.config.hidden_size)]
        else:
            self.ft_linear = [nn.Linear(self.config.dim_ft, self.config.hidden_size)]
        self.ft_linear.append(nn.LayerNorm(self.config.hidden_size))
        self.ft_linear = nn.Sequential(*self.ft_linear)

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, obj_fts):
        '''
        Args:
            obj_fts: LongTensor (batch, num_objs), or, FloatTensor (batch, num_objs, dim_ft)
            obj_locs: FloatTensor (batch, num_objs, dim_loc)
        '''
        obj_embeds = self.ft_linear(obj_fts)
        obj_embeds = self.dropout(obj_embeds)
        return obj_embeds

class PcdObjEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pcd_net = PointNetPP(
            sa_n_points=config.sa_n_points,
            sa_n_samples=config.sa_n_samples,
            sa_radii=config.sa_radii,
            sa_mlps=config.sa_mlps,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, obj_pcds):
        batch_size, num_objs, _, _ = obj_pcds.size()
        # obj_embeds = self.pcd_net(
        #     einops.rearrange(obj_pcds, 'b o p d -> (b o) p d')
        # )
        # obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)

        # TODO: due to the implementation of PointNetPP, this way consumes less GPU memory
        obj_embeds = []
        for i in range(batch_size):
            obj_embeds.append(self.pcd_net(obj_pcds[i]))
        obj_embeds = torch.stack(obj_embeds, 0)

        # obj_embeds = []
        # for i in range(num_objs):
        #     obj_embeds.append(self.pcd_net(obj_pcds[:, i]))
        # obj_embeds = torch.stack(obj_embeds, 1)

        obj_embeds = self.dropout(obj_embeds)
        return obj_embeds


class ObjColorEncoder(nn.Module):
    def __init__(self, hidden_size, dropout=0):
        super().__init__()
        self.ft_linear = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout)
        )

    def forward(self, obj_colors):
        # obj_colors: (batch, nobjs, 3, 4)
        gmm_weights = obj_colors[..., :1]
        gmm_means = obj_colors[..., 1:]

        embeds = torch.sum(self.ft_linear(gmm_means) * gmm_weights, 2)
        return embeds
        