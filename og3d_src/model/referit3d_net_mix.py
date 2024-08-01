import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from transformers import BertConfig, BertModel

from .obj_encoder import GTObjEncoder, PcdObjEncoder
from .mmt_module import MMT
from .cmt_module import CMT
from .referit3d_net import get_mlp_head, freeze_bn
from .referit3d_net import ReferIt3DNet


# 这段代码定义了一个名为ReferIt3DNetMix的PyTorch神经网络模块，
# 它是一个混合模型，用于3D参考解析（3D referential grounding）任务。
# 这个模型结合了一个教师模型和一个学生模型，通过知识蒸馏的方式训练学生模型。
# 以下是ReferIt3DNetMix类的主要组成部分：

# 初始化方法(__init__):
# 根据配置参数，创建教师模型和学生模型。教师模型通常使用更丰富的特征（如类别标签），而学生模型使用更少的信息（如点云数据）。
# 冻结教师模型的参数，使其在训练过程中不更新权重。
# 如果配置参数中指定，设置教师模型的评估模式。

# 准备批次方法(prepare_batch):
# 将批次数据移动到指定的设备（CPU或GPU）。

# 前向传播方法(forward):
# 在评估模式下运行教师模型，以获取参考输出。
# 运行学生模型，并获取其输出。
# 如果需要，计算学生模型的损失。

# 损失计算方法(compute_loss):
# 计算学生模型的标准损失。
# 如果配置参数中指定，计算知识蒸馏的损失，包括注意力矩阵、自注意力矩阵和隐藏状态的差异。
# 将所有损失加总，得到最终的损失。
# 这个模型是专门为3D参考解析任务而设计的，如3D物体检测和3D场景理解。
# 通过使用教师-学生训练策略，学生模型可以从教师模型那里学习更丰富的特征表示，从而在3D理解任务中取得更好的性能。



class ReferIt3DNetMix(nn.Module):
    def __init__(self, config, device):
        # 调用父类的构造函数
        super().__init__()
        # 存储配置信息
        self.config = config
        # 存储设备信息（CPU或GPU）
        self.device = device
        # 获取教师模型评估模式的配置，如果没有设置则默认为False
        self.teacher_eval_mode = config.get('teacher_eval_mode', False)

        # 创建教师模型的配置副本
        teacher_model_cfg = copy.deepcopy(config)
        # 设置教师模型的类型为'gtlabel'
        teacher_model_cfg.model_type = 'gtlabel'
        # 设置教师模型是否使用颜色编码
        teacher_model_cfg.obj_encoder.use_color_enc = teacher_model_cfg.obj_encoder.teacher_use_color_enc

        # 创建教师模型实例
        self.teacher_model = ReferIt3DNet(teacher_model_cfg, device)

        # 创建学生模型的配置副本
        student_model_cfg = copy.deepcopy(config)
        # 设置学生模型的类型为'gtpcd'
        student_model_cfg.model_type = 'gtpcd'
        # 设置学生模型是否使用颜色编码
        student_model_cfg.obj_encoder.use_color_enc = student_model_cfg.obj_encoder.student_use_color_enc

        # 创建学生模型实例
        self.student_model = ReferIt3DNet(student_model_cfg, device)

        # 遍历教师模型的参数，并将它们设置为不需要梯度更新
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def prepare_batch(self, batch):
        # 初始化一个字典用于存储处理后的数据
        outs = {}
        # 遍历输入批次中的每一项
        for key, value in batch.items():
            # 检查数据项是否为torch.Tensor对象
            if isinstance(value, torch.Tensor):
                # 如果是Tensor，则将其转移到指定的设备上（GPU或CPU）
                outs[key] = value.to(self.device)
            else:
                # 如果不是Tensor，则直接使用原始值
                outs[key] = value
        # 返回处理后的数据批次
        return outs

    def forward(self, batch: dict, compute_loss=False, is_test=False) -> dict:
        # 首先调用 prepare_batch 函数处理输入批次，确保所有数据位于正确的设备上
        batch = self.prepare_batch(batch)

        # 如果处于教师模型的评估模式
        if self.teacher_eval_mode:
            # 将教师模型设置为评估模式，禁用梯度计算，适用于推断和验证
            self.teacher_model.eval()
            # 组织教师模型所需的输入数据
            batch_teacher = {
                'obj_fts': batch['obj_gt_fts'],  # 物体特征
                'obj_colors': batch['obj_colors'],  # 物体颜色
                'obj_locs': batch['obj_locs'],  # 物体位置
                'obj_masks': batch['obj_masks'],  # 物体掩码
                'txt_ids': batch['txt_ids'],  # 文本标识
                'txt_masks': batch['txt_masks']  # 文本掩码
            }
            # 使用教师模型进行前向传播，获取输出，不计算损失
            teacher_outs = self.teacher_model(
                batch_teacher, compute_loss=False,
                output_attentions=True, output_hidden_states=True,
            )
            # 对教师模型的输出进行处理，分离梯度
            for k, v in teacher_outs.items():
                if isinstance(v, list):
                    teacher_outs[k] = [x.detach() for x in v]  # 对列表中的每个元素进行分离
                else:
                    teacher_outs[k] = v.detach()  # 分离单个张量

        # 使用学生模型进行前向传播
        student_outs = self.student_model(
            batch, compute_loss=False,
            output_attentions=True, output_hidden_states=True,
        )

        # 如果需要计算损失
        if compute_loss:
            # 调用 compute_loss 方法计算从教师模型到学生模型的知识蒸馏损失
            losses = self.compute_loss(teacher_outs, student_outs, batch)
            # 返回学生模型的输出和计算的损失
            return student_outs, losses

        # 如果不计算损失，仅返回学生模型的输出
        return student_outs

    def compute_loss(self, teacher_outs, student_outs, batch):
        # 首先计算学生模型的基本损失
        losses = self.student_model.compute_loss(student_outs, batch)

        # 如果配置了交叉注意力蒸馏损失并且其权重大于0
        if self.config.losses.distill_cross_attns > 0:
            # 创建交叉注意力的掩码，用于筛选有效的注意力位置
            cross_attn_masks = batch['obj_masks'].unsqueeze(2) * batch['txt_masks'].unsqueeze(1)
            cross_attn_masks = cross_attn_masks.float()
            # 计算掩码的总和，用于后面的平均损失计算
            cross_attn_sum = cross_attn_masks.sum()
            # 遍历每一层的交叉注意力
            for i in range(self.config.mm_encoder.num_layers):
                # 计算教师和学生输出的平方误差
                mse_loss = (teacher_outs['all_cross_attns'][i] - student_outs['all_cross_attns'][i]) ** 2
                # 使用掩码加权后计算平均损失
                mse_loss = torch.sum(mse_loss * cross_attn_masks) / cross_attn_sum
                # 存储每一层的交叉注意力蒸馏损失
                losses['cross_attn_%d' % i] = mse_loss * self.config.losses.distill_cross_attns
                # 更新总损失
                losses['total'] += losses['cross_attn_%d' % i]

        # 如果配置了自注意力蒸馏损失并且其权重大于0
        if self.config.losses.distill_self_attns > 0:
            # 创建自注意力的掩码
            self_attn_masks = batch['obj_masks'].unsqueeze(2) * batch['obj_masks'].unsqueeze(1)
            self_attn_masks = self_attn_masks.float()
            self_attn_sum = self_attn_masks.sum()
            # 遍历每一层的自注意力
            for i in range(self.config.mm_encoder.num_layers):
                mse_loss = (teacher_outs['all_self_attns'][i] - student_outs['all_self_attns'][i]) ** 2
                mse_loss = torch.sum(mse_loss * self_attn_masks) / self_attn_sum
                # 存储每一层的自注意力蒸馏损失
                losses['self_attn_%d' % i] = mse_loss * self.config.losses.distill_self_attns
                # 更新总损失
                losses['total'] += losses['self_attn_%d' % i]

        # 如果配置了隐藏状态蒸馏损失并且其权重大于0
        if self.config.losses.distill_hiddens > 0:
            # 创建隐藏状态的掩码
            hidden_masks = batch['obj_masks'].unsqueeze(2).float()
            hidden_sum = hidden_masks.sum() * self.config.hidden_size
            # 遍历每一层的隐藏状态
            for i in range(self.config.mm_encoder.num_layers + 1):
                mse_loss = (teacher_outs['all_hidden_states'][i] - student_outs['all_hidden_states'][i]) ** 2
                mse_loss = torch.sum(mse_loss * hidden_masks) / hidden_sum
                # 存储每一层的隐藏状态蒸馏损失
                losses['hidden_state_%d' % i] = mse_loss * self.config.losses.distill_hiddens
                # 更新总损失
                losses['total'] += losses['hidden_state_%d' % i]

        # 返回所有损失
        return losses
