import math
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from transformers import BertConfig, BertModel

from .obj_encoder import GTObjEncoder, PcdObjEncoder, ObjColorEncoder
from .txt_encoder import GloveGRUEncoder
from .mmt_module import MMT
from .cmt_module import CMT


# 获取多层感知机头：
# 这个函数接收几个参数来配置多层感知机头的结构和行为：


# 这有助于稳定训练过程。如果dropout大于0，它还会应用dropout（nn.Dropout(dropout)）来减少过拟合。
# 最后，它通过第二个线性层（nn.Linear(hidden_size//2, output_size)）将隐藏层的输出映射到输出空间，得到最终的预测或分类结果。
# 这个多层感知机头可以用于各种任务，比如在3D参考解析任务中，它可以将从3D场景中提取的特征转换为对3D物体的类别预测或位置回归。通过调整input_size、hidden_size和output_size，可以灵活地配置多层感知机头的结构，以满足不同任务的需求。

# 这个函数的功能是创建一个多层感知机的结构和行为，
# 接收参数分别为：输入的神经元数量，隐藏层的神经元数量，输出的神经元数量，应用于隐藏层的dropout比例，dropout=0表示不使用dropout
def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    # Sequential函数接收一个由层组成的列表，允许用户将多个计算层按照顺序组合成一个模型
    return nn.Sequential(
        # 它通过一个线性层将输入特征映射到一个维度为 hidden_size// 2的隐藏层，Linear函数本身是没有hidden_size这个参数的，
        # 这里是在表示第一层的输出是下一个线性层的输入，所以下一个Linear的输入参数也是hidden_size//2(并且向下取整)
                nn.Linear(input_size, hidden_size//2),
                nn.ReLU(),
        # 使用层归一化来对隐藏层的输出进行归一化处理，它主要作用是将指定的数据归一化，即均值为0，方差为1
        # 归一化
                nn.LayerNorm(hidden_size//2, eps=1e-12),
                nn.Dropout(dropout),
                nn.Linear(hidden_size//2, output_size)
            )

def freeze_bn(m):
    '''Freeze BatchNorm Layers'''
    for layer in m.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


class ReferIt3DNet(nn.Module):
    def __init__(self, config, device):
        super().__init__()  # 执行父类nn.Module的初始化操作
        self.config = config  # 存储传入的配置信息
        self.device = device  # 存储设备信息，用于模型的设备部署

        config.obj_encoder.num_obj_classes = config.num_obj_classes  # 设置对象编码器的类别数

        # 根据配置中的模型类型初始化对象编码器
        if self.config.model_type == 'gtlabel':
            self.obj_encoder = GTObjEncoder(config.obj_encoder, config.hidden_size)  # 如果是基于标签的，使用GTObjEncoder
        elif self.config.model_type == 'gtpcd':
            self.obj_encoder = PcdObjEncoder(config.obj_encoder)  # 如果是基于点云的，使用PcdObjEncoder

        # 如果配置要求冻结对象编码器的参数
        if self.config.obj_encoder.freeze:
            freeze_bn(self.obj_encoder)  # 冻结批归一化层
            for p in self.obj_encoder.parameters():
                p.requires_grad = False  # 设置这些参数不参与训练

        # 如果配置要求冻结对象编码器的批归一化层
        if self.config.obj_encoder.freeze_bn:
            freeze_bn(self.obj_encoder)

        # 如果配置中包含使用颜色编码器
        if self.config.obj_encoder.use_color_enc:
            self.obj_color_encoder = ObjColorEncoder(config.hidden_size, config.obj_encoder.dropout)  # 初始化颜色编码器

        # 初始化文本编码器
        if self.config.txt_encoder.type == 'gru':
            self.txt_encoder = GloveGRUEncoder(config.hidden_size,
                                               config.txt_encoder.num_layers)  # 如果是GRU类型，使用GloveGRUEncoder
        else:
            txt_bert_config = BertConfig(
                hidden_size=config.hidden_size,
                num_hidden_layers=config.txt_encoder.num_layers,
                num_attention_heads=12, type_vocab_size=2
            )
            self.txt_encoder = BertModel.from_pretrained(
                r'/root/vil3dref-main/moxing/bert-base-uncased/pytorch_model.bin', config=txt_bert_config
            )  # 否则，使用预训练的BERT模型

        # 如果配置要求冻结文本编码器的参数
        if self.config.txt_encoder.freeze:
            for p in self.txt_encoder.parameters():
                p.requires_grad = False  # 设置这些参数不参与训练

        # 初始化多模态编码器的配置
        mm_config = EasyDict(config.mm_encoder)
        mm_config.hidden_size = config.hidden_size
        mm_config.num_attention_heads = 12
        mm_config.dim_loc = config.obj_encoder.dim_loc

        # 根据配置选择多模态编码器的类型
        if self.config.mm_encoder.type == 'cmt':
            self.mm_encoder = CMT(mm_config)  # 如果是CMT类型，初始化CMT编码器
        elif self.config.mm_encoder.type == 'mmt':
            self.mm_encoder = MMT(mm_config)  # 如果是MMT类型，初始化MMT编码器

        # 初始化多层感知机头部用于3D检测
        self.og3d_head = get_mlp_head(
            config.hidden_size, config.hidden_size,
            1, dropout=config.dropout
        )

        # 如果配置中包含对象3D分类损失
        if self.config.losses.obj3d_clf > 0:
            self.obj3d_clf_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                config.num_obj_classes, dropout=config.dropout
            )

        # 如果配置中包含对象3D预分类损失
        if self.config.losses.obj3d_clf_pre > 0:
            self.obj3d_clf_pre_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                config.num_obj_classes, dropout=config.dropout
            )

        # 如果配置要求冻结对象编码器
        if self.config.obj_encoder.freeze:
            for p in self.obj3d_clf_pre_head.parameters():
                p.requires_grad = False  # 设置这些参数不参与训练

        # 如果配置中包含对象3D定位损失
        if self.config.losses.obj3d_reg > 0:
            self.obj3d_reg_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                3, dropout=config.dropout
            )

        # 如果配置中包含文本分类损失
        if self.config.losses.txt_clf > 0:
            self.txt_clf_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                config.num_obj_classes, dropout=config.dropout
            )

    def prepare_batch(self, batch):
        outs = {}  # 初始化一个字典，用来存储处理后的数据
        for key, value in batch.items():  # 遍历批次数据中的每个项
            if isinstance(value, torch.Tensor):  # 检查值是否为张量
                outs[key] = value.to(self.device)  # 如果是张量，将其转移到指定的设备（如GPU）
            else:
                outs[key] = value  # 如果不是张量，直接保留原值
        return outs  # 返回处理后的批次数据字典

    def forward(
            self, batch: dict, compute_loss=False, is_test=False,
            output_attentions=False, output_hidden_states=False,
    ) -> dict:
        batch = self.prepare_batch(batch)  # 预处理批次数据，确保所有数据在正确的设备上
        if self.config.obj_encoder.freeze or self.config.obj_encoder.freeze_bn:
            freeze_bn(self.obj_encoder)  # 如果需要，冻结对象编码器的批量归一化

        obj_embeds = self.obj_encoder(batch['obj_fts'])  # 从对象特征获取对象嵌入
        if self.config.obj_encoder.freeze:
            obj_embeds = obj_embeds.detach()  # 如果对象编码器被冻结，防止梯度回传

        if self.config.obj_encoder.use_color_enc:
            obj_embeds = obj_embeds + self.obj_color_encoder(batch['obj_colors'])  # 如果使用颜色编码器，加入颜色嵌入

        txt_embeds = self.txt_encoder(
            batch['txt_ids'], batch['txt_masks'],
        ).last_hidden_state  # 从文本编码器获取文本嵌入

        if self.config.txt_encoder.freeze:
            txt_embeds = txt_embeds.detach()  # 如果文本编码器被冻结，防止梯度回传

        out_embeds = self.mm_encoder(
            txt_embeds, batch['txt_masks'],
            obj_embeds, batch['obj_locs'], batch['obj_masks'],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )  # 使用多模态编码器处理嵌入和注意力输出

        og3d_logits = self.og3d_head(out_embeds['obj_embeds']).squeeze(2)  # 使用多层感知机头部获取对象检测逻辑回归输出
        og3d_logits.masked_fill_(batch['obj_masks'].logical_not(), -float('inf'))  # 掩盖无效对象的输出

        result = {'og3d_logits': og3d_logits}  # 存储对象检测结果

        if output_attentions:
            result['all_cross_attns'] = out_embeds['all_cross_attns']  # 如果需要输出注意力，添加交叉注意力矩阵
            result['all_self_attns'] = out_embeds['all_self_attns']  # 添加自注意力矩阵

        if output_hidden_states:
            result['all_hidden_states'] = out_embeds['all_hidden_states']  # 如果需要输出隐藏状态，添加所有隐藏状态

        if self.config.losses.obj3d_clf > 0:
            result['obj3d_clf_logits'] = self.obj3d_clf_head(out_embeds['obj_embeds'])  # 如果有对象分类任务，获取分类结果

        if self.config.losses.obj3d_reg > 0:
            result['obj3d_loc_preds'] = self.obj3d_reg_head(out_embeds['obj_embeds'])  # 如果有对象位置回归任务，获取位置预测

        if self.config.losses.obj3d_clf_pre > 0:
            result['obj3d_clf_pre_logits'] = self.obj3d_clf_pre_head(obj_embeds)  # 如果有对象预分类任务，获取预分类结果

        if self.config.losses.txt_clf > 0:
            result['txt_clf_logits'] = self.txt_clf_head(txt_embeds[:, 0])  # 如果有文本分类任务，获取文本分类结果

        if compute_loss:
            losses = self.compute_loss(result, batch)  # 如果需要计算损失，执行损失计算
            return result, losses  # 返回结果和损失

        return result  # 如果不计算损失，只返回结果

    # 这个函数compute_loss是用于计算模型在给定批次数据上的损失。
    # 它接收两个参数：result和batch。result包含了模型的输出，而batch包含了与这些输出相对应的真实标签。
    def compute_loss(self, result, batch):
        # 首先初始化一个字典losses来存储不同类型的损失，以及一个变量total_loss来存储所有损失的总和。
        losses = {}
        total_loss = 0

        # 3D物体检测损失（og3d_loss）：使用交叉熵损失函数（F.cross_entropy）计算3D物体检测的损失。
        # 这个损失是 “模型对3D物体类别的预测” 和 “真实类别标签” 之间的差异。
        # 计算出的损失被添加到losses字典中，并累加到total_loss。
        # result['og3d_logits']是模型的输出，通常是一个多维张量，其中每一行代表一个样本，每一列代表一个类别预测的原始分数
        # batch['tgt_obj_idxs']这是真实标签，它告诉模型每个样本的正确类别。这个张量中的每个元素是一个整数，对应于每个样本的正确类别索引
        # 至于是不是0-1的独热编码我现在还没看懂
        og3d_loss = F.cross_entropy(result['og3d_logits'], batch['tgt_obj_idxs'])
        losses['og3d'] = og3d_loss
        total_loss += og3d_loss

        # 3D物体分类损失（obj3d_clf_loss）：
        # 如果配置参数中指定了obj3d_clf损失，则计算3D物体的分类损失。
        # 使用交叉熵损失函数，但这次考虑了每个物体的损失，并且只对非填充物体（batch['obj_masks']）计算损失。
        # 计算出的损失乘以配置参数中的权重，然后添加到losses字典和total_loss。
        if self.config.losses.obj3d_clf > 0:
            obj3d_clf_loss = F.cross_entropy(
                result['obj3d_clf_logits'].permute(0, 2, 1), 
                batch['obj_classes'], reduction='none'
            )
            obj3d_clf_loss = (obj3d_clf_loss * batch['obj_masks']).sum() / batch['obj_masks'].sum()
            losses['obj3d_clf'] = obj3d_clf_loss * self.config.losses.obj3d_clf
            total_loss += losses['obj3d_clf']

        # 3D物体分类预测损失（Obj3D CLF Pre Loss）：
        # 如果配置参数中指定了obj3d_clf_pre损失，则计算3D物体的分类预测损失。
        # 这个损失的计算方式与3D物体分类损失相同，但是基于不同的模型输出（result['obj3d_clf_pre_logits']）。
        # 同样，计算出的损失乘以配置参数中的权重，然后添加到losses字典和total_loss。
        if self.config.losses.obj3d_clf_pre > 0:
            obj3d_clf_pre_loss = F.cross_entropy(
                result['obj3d_clf_pre_logits'].permute(0, 2, 1), 
                batch['obj_classes'], reduction='none'
            )
            obj3d_clf_pre_loss = (obj3d_clf_pre_loss * batch['obj_masks']).sum() / batch['obj_masks'].sum()
            losses['obj3d_clf_pre'] = obj3d_clf_pre_loss * self.config.losses.obj3d_clf_pre
            total_loss += losses['obj3d_clf_pre']

        # 3D物体定位损失（Obj3D REG Loss）：
        # 如果配置参数中指定了obj3d_reg损失，则计算3D物体的定位损失。
        # 使用均方误差损失函数（F.mse_loss）来衡量模型预测的物体位置（result['obj3d_loc_preds']）和真实物体位置（batch['obj_locs'][:, :, :3]）之间的差异。
        # 同样，只对非填充物体计算损失，并且计算出的损失乘以配置参数中的权重，然后添加到losses字典和total_loss。
        if self.config.losses.obj3d_reg > 0:
            obj3d_reg_loss = F.mse_loss(
                result['obj3d_loc_preds'], batch['obj_locs'][:, :, :3],  reduction='none'
            )
            obj3d_reg_loss = (obj3d_reg_loss * batch['obj_masks'].unsqueeze(2)).sum() / batch['obj_masks'].sum()
            losses['obj3d_reg'] = obj3d_reg_loss * self.config.losses.obj3d_reg
            total_loss += losses['obj3d_reg']

        # 文本分类损失（Txt CLF Loss）：
        # 如果配置参数中指定了txt_clf损失，则计算文本分类损失。
        # 使用交叉熵损失函数，基于模型对文本类别的预测（result['txt_clf_logits']）和真实文本类别（batch['tgt_obj_classes']）之间的差异。
        # 计算出的损失乘以配置参数中的权重，然后添加到losses字典和total_loss。
        # 最后，total_loss被添加到losses字典中，标记为'total'。这个字典包含了所有单独的损失以及它们的总和，然后返回给调用者。
        if self.config.losses.txt_clf > 0:
            txt_clf_loss = F.cross_entropy(
                result['txt_clf_logits'], batch['tgt_obj_classes'],  reduction='mean'
            )
            losses['txt_clf'] = txt_clf_loss * self.config.losses.txt_clf
            total_loss += losses['txt_clf']

        losses['total'] = total_loss
        return losses
