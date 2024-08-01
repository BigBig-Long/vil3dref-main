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
        return F.relu  # 如果激活函数类型是 'relu'，则返回 PyTorch 的 relu 函数

    if activation == "gelu":
        return F.gelu  # 如果激活函数类型是 'gelu'，则返回 PyTorch 的 gelu 函数

    if activation == "glu":
        return F.glu  # 如果激活函数类型是 'glu'，则返回 PyTorch 的 glu 函数

    # 如果提供的激活类型不是支持的类型，则抛出运行时错误
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# 这个函数的目的是创建一个包含指定数量（N）个模块副本的nn.ModuleList
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    # 使用列表推导式来创建N个深拷贝的module实例，并将这些实例封装到nn.ModuleList中
    # nn.ModuleList是PyTorch中的一个容器类，可以存储多个模块，并自动处理它们的参数

# Transformer编码层实现
class TransformerEncoderLayer(nn.Module):
    def __init__(
            self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
            activation="relu"
    ):
        super().__init__()  # 调用父类的构造函数，初始化模块的基本结构
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 实例化多头注意力模块，d_model是模型的维度，nhead是头的数量，dropout是随机丢弃比率，batch_first指定输入的第一维是批次大小

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # 实例化第一个线性层，输入维度为d_model，输出维度为dim_feedforward

        self.dropout = nn.Dropout(dropout)
        # 实例化dropout层，dropout比率为传入的dropout参数

        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 实例化第二个线性层，输入维度为dim_feedforward，输出维度回到d_model

        self.norm1 = nn.LayerNorm(d_model)
        # 实例化第一个层归一化层，归一化处理的维度为d_model

        self.norm2 = nn.LayerNorm(d_model)
        # 实例化第二个层归一化层，归一化处理的维度为d_model

        self.dropout1 = nn.Dropout(dropout)
        # 实例化第一个dropout层，用于第一层归一化后的处理

        self.dropout2 = nn.Dropout(dropout)
        # 实例化第二个dropout层，用于第二层归一化后的处理

        self.activation = _get_activation_fn(activation)
        # 根据传入的激活函数名称获取对应的激活函数

    def forward(
            self, src, src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        # 对输入数据 src 应用层归一化

        src2 = self.self_attn(
            src2, src2, value=src2, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        # 应用多头注意力机制，src2作为query, key和value, 使用可选的注意力掩码和键掩码

        src = src + self.dropout1(src2)
        # 将注意力机制的输出经过dropout后与原始输入进行残差连接

        src2 = self.norm2(src)
        # 对残差连接的结果再次进行层归一化

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # 通过第一个线性层，应用激活函数，再通过dropout，最后通过第二个线性层

        src = src + self.dropout2(src2)
        # 将线性层的输出经过dropout后与之前的残差连接结果进行第二次残差连接

        return src
        # 返回这一层的输出，为下一层或输出提供数据

    def forward_post(
            self, src, src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.self_attn(
            src, src, value=src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        # 应用多头注意力机制，src 作为 query, key 和 value，使用可选的注意力掩码和键掩码

        src = src + self.dropout1(src2)
        # 将注意力机制的输出经过 dropout 后与原始输入进行残差连接

        src = self.norm1(src)
        # 对残差连接的结果应用层归一化

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 通过第一个线性层，应用激活函数，再通过 dropout，最后通过第二个线性层

        src = src + self.dropout2(src2)
        # 将线性层的输出经过 dropout 后与之前的残差连接结果进行第二次残差连接

        src = self.norm2(src)
        # 对第二次残差连接的结果应用层归一化

        return src
        # 返回这一层的输出，为下一层或输出提供数据


class MMT(nn.Module):

    def __init__(self, config):
        super().__init__()  # 调用父类的构造函数进行初始化
        self.config = config  # 保存配置对象，用于后续使用其中的配置参数

        # 创建一个 Transformer 编码层，配置其参数
        decoder_layer = TransformerEncoderLayer(
            config.hidden_size,  # 隐藏层的大小
            config.num_attention_heads,  # 注意力机制的头数
            dim_feedforward=2048,  # 前馈网络中间层的维度
            dropout=0.1,  # dropout比率
            activation='gelu'  # 激活函数
        )

        # 克隆编码层，创建多个相同的层
        self.layers = _get_clones(decoder_layer, config.num_hidden_layers)  # 复制这个层，根据隐藏层数量

        # 定义位置编码层，将位置维度映射到隐藏层维度
        loc_layer = nn.Sequential(
            nn.Linear(config.dim_loc, config.hidden_size),  # 线性层，从位置维度到隐藏层维度
            nn.LayerNorm(config.hidden_size)  # 层归一化
        )

        # 根据配置决定如何克隆位置编码层
        if self.config.obj_loc_encoding in ['same_0', 'same_all']:  # 如果位置编码方式为同一编码
            num_loc_layers = 1  # 只需一层
        elif self.config.obj_loc_encoding == 'diff_all':  # 如果每层的位置编码都不同
            num_loc_layers = config.num_hidden_layers  # 则每层都需要一个位置编码层

        # 克隆位置编码层
        self.loc_layers = _get_clones(loc_layer, num_loc_layers)  # 克隆位置层，根据需要的层数

        # 应用权重初始化函数
        self.apply(self._init_weights)  # 使用自定义的权重初始化方法初始化模型权重

    def _init_weights(self, module):
        """Initialize the weights"""
        # 检查是否是线性层
        if isinstance(module, nn.Linear):
            # 初始化权重为均值为0，标准差为0.02的正态分布
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 如果偏置存在，初始化偏置为0
            if module.bias is not None:
                module.bias.data.zero_()

        # 检查是否是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层权重为均值为0，标准差为0.02的正态分布
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 如果存在填充索引，将该索引位置的权重置零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        # 检查是否是层归一化
        elif isinstance(module, nn.LayerNorm):
            # 将层归一化的偏置初始化为0
            module.bias.data.zero_()
            # 将层归一化的权重初始化为1
            module.weight.data.fill_(1.0)

    def forward(
            self, txt_embeds, txt_masks, obj_embeds, obj_locs, obj_masks,
            output_attentions=False, output_hidden_states=False,
    ):
        # 获取文本和对象嵌入的最大长度
        max_txt_len = txt_embeds.size(1)
        max_obj_len = obj_embeds.size(1)

        # 将文本和对象嵌入向量拼接
        hidden_states = torch.cat([txt_embeds, obj_embeds], dim=1)

        # 合并文本和对象的掩码，并进行逻辑非操作，为后续注意力层准备
        padding_masks = torch.cat([txt_masks, obj_masks], dim=1).logical_not()

        # 初始化隐藏状态列表，用于存储每层的输出
        all_hidden_states = [hidden_states]

        # 遍历每一层
        for i, layer in enumerate(self.layers):
            # 根据拼接后的隐藏状态和文本长度，分离文本和对象的嵌入
            txt_embeds = hidden_states[:, :max_txt_len]
            obj_embeds = hidden_states[:, max_txt_len:]

            # 根据配置，决定如何处理对象的位置信息
            if self.config.obj_loc_encoding == 'diff_all':
                # 如果每层的位置编码都不同，从位置层集合中获取当前层对应的位置编码
                new_obj_locs = self.loc_layers[i](obj_locs)
                obj_embeds = obj_embeds + new_obj_locs
            else:
                # 否则，使用第一层的位置编码
                new_obj_locs = self.loc_layers[0](obj_locs)
                if self.config.obj_loc_encoding == 'same_all':
                    # 如果所有层的位置编码相同
                    obj_embeds = obj_embeds + new_obj_locs
                else:
                    # 如果只有第一层使用位置编码
                    if i == 0:
                        obj_embeds = obj_embeds + new_obj_locs

            # 重新拼接更新后的文本和对象嵌入
            hidden_states = torch.cat([txt_embeds, obj_embeds], dim=1)

            # 应用当前层的处理
            hidden_states = layer(
                hidden_states,
                src_key_padding_mask=padding_masks,
            )

            # 将这一层的隐藏状态加入到列表中
            all_hidden_states.append(hidden_states)

        # 构建输出字典，包括最终的文本和对象嵌入
        outs = {
            'txt_embeds': hidden_states[:, :max_txt_len],
            'obj_embeds': hidden_states[:, max_txt_len:],
        }

        # 如果请求输出隐藏状态，添加到输出字典中
        if output_hidden_states:
            outs['all_hidden_states'] = all_hidden_states

        # 返回最终的输出字典
        return outs


