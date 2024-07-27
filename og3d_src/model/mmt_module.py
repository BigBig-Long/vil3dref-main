from typing import Optional

import einops
import copy

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# 这段代码定义了一个名为MMT（可能代表“多模态Transformer”）的PyTorch神经网络模块，
# 它是一个基于Transformer的编码器模型，专门用于处理多模态数据，包括文本和对象嵌入。
# 这个模型通过结合Transformer的强大建模能力，能够有效地处理和融合不同模态的数据。
# 以下是MMT类的主要组成部分：
# 初始化方法(__init__):
# 根据配置参数创建Transformer编码器层。
# 创建位置编码层，用于处理对象的位置信息。
# 根据配置参数决定是否为每一层都创建一个位置编码层，或者所有层共享一个位置编码层。
# 初始化权重。
# 权重初始化方法(_init_weights):
# 初始化线性层、嵌入层和层规范化的权重。
# 前向传播方法(forward):
# 接受文本嵌入、文本掩码、对象嵌入、对象位置和对象掩码作为输入。
# 将文本嵌入和对象嵌入拼接在一起，并创建相应的padding掩码。
# 通过编码器层逐层处理拼接后的嵌入，结合对象的位置信息。
# 如果需要，输出所有隐藏状态。
# 这个模型是专门为处理多模态数据而设计的，如文本描述和3D点云数据的结合。
# 它通过Transformer编码器层有效地融合了文本和空间信息，可以用于各种多模态任务，如3D物体检测或3D场景理解。


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

# Transformer编码层实现
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
        activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self, src, src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        src2 = self.self_attn(
            src2, src2, value=src2, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward_post(
        self, src, src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.self_attn(
            src, src, value=src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MMT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        decoder_layer = TransformerEncoderLayer(
            config.hidden_size, config.num_attention_heads,
            dim_feedforward=2048, dropout=0.1, activation='gelu'
        )
        self.layers = _get_clones(decoder_layer, config.num_hidden_layers)

        loc_layer = nn.Sequential(
            nn.Linear(config.dim_loc, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )
        if self.config.obj_loc_encoding in ['same_0', 'same_all']:
            num_loc_layers = 1
        elif self.config.obj_loc_encoding == 'diff_all':
            num_loc_layers = config.num_hidden_layers
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

    def forward(
        self, txt_embeds, txt_masks, obj_embeds, obj_locs, obj_masks,
        output_attentions=False, output_hidden_states=False,
    ):
        max_txt_len = txt_embeds.size(1)
        max_obj_len = obj_embeds.size(1)

        hidden_states = torch.cat([txt_embeds, obj_embeds], dim=1)
        padding_masks = torch.cat([txt_masks, obj_masks], dim=1).logical_not()
        all_hidden_states = [hidden_states]
        for i, layer in enumerate(self.layers):
            txt_embeds = hidden_states[:, :max_txt_len]
            obj_embeds = hidden_states[:, max_txt_len:]

            if self.config.obj_loc_encoding == 'diff_all':
                new_obj_locs = self.loc_layers[i](obj_locs)
                obj_embeds = obj_embeds + new_obj_locs
            else:
                new_obj_locs = self.loc_layers[0](obj_locs)
                if self.config.obj_loc_encoding == 'same_all':
                    obj_embeds = obj_embeds + new_obj_locs
                else:
                    if i == 0:
                        obj_embeds = obj_embeds + new_obj_locs

            hidden_states = torch.cat([txt_embeds, obj_embeds], dim=1)

            hidden_states = layer(
                hidden_states,
                src_key_padding_mask=padding_masks,
            )
            all_hidden_states.append(hidden_states)

        outs = {
            'txt_embeds': hidden_states[:, :max_txt_len],
            'obj_embeds': hidden_states[:, max_txt_len:],
        }
        if output_hidden_states:
            outs['all_hidden_states'] = all_hidden_states
        return outs

