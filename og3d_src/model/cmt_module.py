import copy
import numpy as np
from typing import Optional
import time
import einops
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# 这段代码定义了一个名为CMT（可能代表“组合模型Transformer”）的PyTorch神经网络模块，
# 它是一个基于Transformer的编码器-解码器模型，专门用于处理空间信息。
# 这个模型结合了Transformer解码器层和空间注意力机制，用于处理点云数据和文本描述。
# 以下是CMT类的主要组成部分：

# 位置编码计算方法(calc_pairwise_locs):
# 计算对象之间的成对位置关系。
# 支持多种类型的成对位置关系计算，如基于中心的距离、基于底部的距离等。
# 前向传播方法(forward):
# 接受文本嵌入、文本掩码、对象嵌入、对象位置和对象掩码作为输入。
# 计算对象之间的成对位置关系。
# 通过解码器层逐层处理对象嵌入，结合文本嵌入和位置信息。
# 如果需要，输出所有隐藏状态和注意力矩阵。
# 这个模型是专门为处理3D场景理解任务而设计的，如3D物体检测或3D场景分割。
# 它通过结合Transformer的强大建模能力和空间注意力机制，能够有效地处理点云数据和文本描述，从而在3D理解任务中取得更好的性能。

# 这个函数接收一个字符串参数，根据接收的字符串返回对应的激活函数，如果没有选择指定的函数，会抛出异常
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# 这个函数的目的是创建一个包含指定数量（N）个模块副本的nn.ModuleList
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Transformer解码层实现
# nn.Module 是所有神经网络模块的基类。模型都应该继承这个类，并实现 forward 方法。
class TransformerDecoderLayer(nn.Module):

    # 初始化方法(__init__):
    # d_model：表示模型的输入总维度(官方给出的解释是Total dimension of the model，即模型的输入总维度，每个头输入的维度×头的数量)
    # nhead： 表示多头注意力机制中的头的数量。
    # dim_feedforward： 表示前馈网络中隐藏层的维度，默认为2048。
    # dropout=0.1 ： 表示应用于各种层的dropout比例，默认为 0.1。
    # activation="relu"：表示前馈网络中使用的激活函数，默认为"relu"。
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        # 网络结构
        super().__init__()
        # 下面使用 MultiheadAttention函数 创建了两个多头注意力模块“self.self_attn”和“self.multihead_attn”
        # 它们用于在编码器层中处理自注意力（self - attention）和交叉注意力（cross - attention）。
        # 接收 嵌入维度、头数、dropout比例（防止过拟合）、batch_first=True的意思是

        # 当batch_first = True时，输入和输出的张量形状将是(batch_size, sequence_length,feature_dim)，
        # 其中batch_size是批量大小，sequence_length是序列的长度（比如句子的单词数），feature_dim是特征维度（比如词向量的维度）。
        # 例如，对于一个句子批量的输入，如果batch_first = True，张量的形状将是(batch_size, seq_len,
        # embedding_dim)，其中batch_size是批量的句子数量，seq_len是每个句子的单词数量，embedding_dim是每个单词的嵌入维度。
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model(前馈模型的实现)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    # 前向传播，
    # 这个函数接收几个参数，这些参数代表了Transformer层在处理序列数据时的输入和掩码：
    # tgt：目标序列的表示，通常是上一个时间步的输出或解码器层的输入。
    # memory：记忆序列的表示，通常是编码器层的输出，用于解码器层的交叉注意力机制。
    # tgt_mask：目标序列的注意力掩码，用于屏蔽未来的位置。
    # memory_mask：记忆序列的注意力掩码，用于屏蔽某些位置的注意力。
    # tgt_key_padding_mask：目标序列的键填充掩码，用于屏蔽填充位置的注意力。
    # memory_key_padding_mask：记忆序列的键填充掩码，用于屏蔽填充位置的注意力。
    # 在函数内部，首先对目标序列tgt进行归一化处理（self.norm1(
    #     tgt)）。然后，使用自注意力机制（self.self_attn）对归一化后的目标序列进行注意力计算，得到注意力矩阵和更新后的目标序列tgt2。接着，将更新后的目标序列与原始目标序列相加，并应用dropout（self.dropout1(
    #     tgt2)）。
    #
    # 然后，对更新后的目标序列再次进行归一化处理（self.norm2(
    #     tgt)），并使用交叉注意力机制（self.multihead_attn）对目标序列和记忆序列进行注意力计算，得到注意力矩阵和更新后的目标序列tgt2。再次将更新后的目标序列与原始目标序列相加，并应用dropout（self.dropout2(
    #     tgt2)）。
    # 接下来，对更新后的目标序列进行第三次归一化处理（self.norm3(
    #     tgt)），并通过前馈网络（包括两个线性层和激活函数）进行处理，得到更新后的目标序列tgt2。最后，将更新后的目标序列与原始目标序列相加，并应用dropout（self.dropout3(
    #     tgt2)）。
    # 最终，函数返回更新后的目标序列tgt，以及自注意力和交叉注意力的注意力矩阵。
    # 这个函数是Transformer模型中非常核心的部分，它通过自注意力和交叉注意力机制，以及前馈网络，来更新序列的表示，从而实现对序列数据的深层次理解和转换。
    def forward(
        self, tgt, memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices

# 这个模块的目的是在Transformer模型中实现多头注意力机制，并且将空间信息整合到注意力计算中
class MultiHeadAttentionSpatial(nn.Module):
    # 这里相对于
    def __init__(
        self, d_model, n_head, dropout=0.1, spatial_multihead=True, spatial_dim=5,
        spatial_attn_fusion='mul',
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' %(d_model, n_head)

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_multihead = spatial_multihead
        self.spatial_dim = spatial_dim
        self.spatial_attn_fusion = spatial_attn_fusion

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.spatial_n_head = n_head if spatial_multihead else 1
        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)
        elif self.spatial_attn_fusion == 'ctx':
            self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)
        elif self.spatial_attn_fusion == 'cond':
            self.lang_cond_fc = nn.Linear(d_model, self.spatial_n_head * (spatial_dim + 1))
        else:
            raise NotImplementedError('unsupported spatial_attn_fusion %s' % (self.spatial_attn_fusion))


    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
        residual = q
        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t h -> h b l t') 
            if self.spatial_attn_fusion == 'mul':
                loc_attn = F.relu(loc_attn)
            if not self.spatial_multihead:
                loc_attn = einops.repeat(loc_attn, 'h b l t -> (h nh) b l t', nh=self.n_head)
        elif self.spatial_attn_fusion == 'ctx':
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t (h k) -> h b l t k', h=self.n_head)
            loc_attn = torch.einsum('hblk,hbltk->hblt', q, loc_attn) / np.sqrt(q.shape[-1])
        elif self.spatial_attn_fusion == 'cond':
            spatial_weights = self.lang_cond_fc(residual + txt_embeds.unsqueeze(1))
            spatial_weights = einops.rearrange(spatial_weights, 'b l (h d) -> h b l d', h=self.spatial_n_head, d=self.spatial_dim+1)
            if self.spatial_n_head == 1:
                spatial_weights = einops.repeat(spatial_weights, '1 b l d -> h b l d', h=self.n_head)
            spatial_bias = spatial_weights[..., :1]
            spatial_weights = spatial_weights[..., 1:]
            loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights, pairwise_locs) + spatial_bias
            loc_attn = torch.sigmoid(loc_attn)

        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)
            if self.spatial_attn_fusion in ['mul', 'cond']:
                loc_attn = loc_attn.masked_fill(mask, 0)
            else:
                loc_attn = loc_attn.masked_fill(mask, -np.inf)

        if self.spatial_attn_fusion == 'add':
            fused_attn = (torch.softmax(attn, 3) + torch.softmax(loc_attn, 3)) / 2
        else:
            if self.spatial_attn_fusion in ['mul', 'cond']:
                fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
            else:
                fused_attn = loc_attn + attn
            fused_attn = torch.softmax(fused_attn, 3)
        
        assert torch.sum(torch.isnan(fused_attn) == 0), print(fused_attn)

        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, fused_attn


class TransformerSpatialDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
        spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        super().__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        del self.self_attn
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout, 
            spatial_multihead=spatial_multihead, 
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

    def forward(
        self, tgt, memory, tgt_pairwise_locs,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):

        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, tgt2, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask,
            txt_embeds=memory[:, 0],
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices


class CMT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.spatial_dec:
            decoder_class = TransformerSpatialDecoderLayer
            kwargs = {
                'spatial_dim': config.spatial_dim,
                'spatial_multihead': config.spatial_multihead,
                'spatial_attn_fusion': config.spatial_attn_fusion,
            }
        else:
            decoder_class = TransformerDecoderLayer
            kwargs = {}

        decoder_layer = decoder_class(
            config.hidden_size, config.num_attention_heads,
            dim_feedforward=2048, dropout=0.1, activation='gelu', **kwargs
        )
        self.layers = _get_clones(decoder_layer, config.num_layers)

        loc_layer = nn.Sequential(
            nn.Linear(config.dim_loc, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )
        if self.config.obj_loc_encoding in ['same_0', 'same_all']:
            num_loc_layers = 1
        elif self.config.obj_loc_encoding == 'diff_all':
            num_loc_layers = config.num_layers
        self.loc_layers = _get_clones(loc_layer, num_loc_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def calc_pairwise_locs(self, obj_centers, obj_whls, eps=1e-10, pairwise_rel_type='center'):
        if pairwise_rel_type == 'mlp':
            obj_locs = torch.cat([obj_centers, obj_whls], 2)
            pairwise_locs = torch.cat(
                [einops.repeat(obj_locs, 'b l d -> b l x d', x=obj_locs.size(1)),
                einops.repeat(obj_locs, 'b l d -> b x l d', x=obj_locs.size(1))],
                dim=3
            )
            return pairwise_locs

        pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
            - einops.repeat(obj_centers, 'b l d -> b 1 l d')
        pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, 3) + eps) # (b, l, l)
        if self.config.spatial_dist_norm:
            max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
            norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, 'b -> b 1 1')
        else:
            norm_pairwise_dists = pairwise_dists

        if self.config.spatial_dim == 1:
            return norm_pairwise_dists.unsqueeze(3)
            
        pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2]**2, 3)+eps)
        if pairwise_rel_type == 'center':
            pairwise_locs = torch.stack(
                [norm_pairwise_dists, pairwise_locs[..., 2]/pairwise_dists, 
                pairwise_dists_2d/pairwise_dists, pairwise_locs[..., 1]/pairwise_dists_2d,
                pairwise_locs[..., 0]/pairwise_dists_2d],
                dim=3
            )
        elif pairwise_rel_type == 'vertical_bottom':
            bottom_centers = torch.clone(obj_centers)
            bottom_centers[:, :, 2] -= obj_whls[:, :, 2]
            bottom_pairwise_locs = einops.repeat(bottom_centers, 'b l d -> b l 1 d') \
                - einops.repeat(bottom_centers, 'b l d -> b 1 l d')
            bottom_pairwise_dists = torch.sqrt(torch.sum(bottom_pairwise_locs**2, 3) + eps) # (b, l, l)
            bottom_pairwise_dists_2d = torch.sqrt(torch.sum(bottom_pairwise_locs[..., :2]**2, 3)+eps)
            pairwise_locs = torch.stack(
                [norm_pairwise_dists, 
                bottom_pairwise_locs[..., 2]/bottom_pairwise_dists, 
                bottom_pairwise_dists_2d/bottom_pairwise_dists, 
                pairwise_locs[..., 1]/pairwise_dists_2d,
                pairwise_locs[..., 0]/pairwise_dists_2d],
                dim=3
            )
            
        if self.config.spatial_dim == 4:
            pairwise_locs = pairwise_locs[..., 1:]
        return pairwise_locs

    def forward(
        self, txt_embeds, txt_masks, obj_embeds, obj_locs, obj_masks,
        output_attentions=False, output_hidden_states=False, 
    ):
        if self.config.spatial_dec:
            pairwise_locs = self.calc_pairwise_locs(
                obj_locs[:, :, :3], obj_locs[:, :, 3:], 
                pairwise_rel_type=self.config.pairwise_rel_type
            )

        out_embeds = obj_embeds
        all_hidden_states = [out_embeds]
        all_self_attn_matrices, all_cross_attn_matrices = [], []
        for i, layer in enumerate(self.layers):
            if self.config.obj_loc_encoding == 'diff_all':
                query_pos = self.loc_layers[i](obj_locs)
                out_embeds = out_embeds + query_pos
            else:
                query_pos = self.loc_layers[0](obj_locs)
                if self.config.obj_loc_encoding == 'same_all':
                    out_embeds = out_embeds + query_pos
                else:
                    if i == 0:
                        out_embeds = out_embeds + query_pos

            if self.config.spatial_dec:
                out_embeds, self_attn_matrices, cross_attn_matrices = layer(
                    out_embeds, txt_embeds, pairwise_locs,
                    tgt_key_padding_mask=obj_masks.logical_not(),
                    memory_key_padding_mask=txt_masks.logical_not(),
                )
            else:
                out_embeds, self_attn_matrices, cross_attn_matrices = layer(
                    out_embeds, txt_embeds,
                    tgt_key_padding_mask=obj_masks.logical_not(),
                    memory_key_padding_mask=txt_masks.logical_not(),
                )

            all_hidden_states.append(out_embeds)
            all_self_attn_matrices.append(self_attn_matrices)
            all_cross_attn_matrices.append(cross_attn_matrices)

        outs = {
            'obj_embeds': out_embeds,
        }
        if output_hidden_states:
            outs['all_hidden_states'] = all_hidden_states
        if output_attentions:
            outs['all_self_attns'] = all_self_attn_matrices
            outs['all_cross_attns'] = all_cross_attn_matrices
        return outs
