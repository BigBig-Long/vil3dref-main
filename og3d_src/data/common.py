import torch
from einops import repeat

# 定义一个函数用来生成序列掩码
def gen_seq_masks(seq_lens, max_len=None):
    """
    参数:
    seq_lens: torch.LongTensor, 形状=(N, )，表示每个序列的长度
    返回:
    masks: torch.BoolTensor, 形状=(N, L), 其中未达到序列长度的位置为0
    """
    # 如果没有指定最大长度，则计算最大长度
    if max_len is None:
        max_len = max(seq_lens)
    # 获取序列批次的大小
    batch_size = len(seq_lens)
    # 生成一个0到max_len的序列，并将其复制到每个批次
    seq_masks = repeat(torch.arange(max_len).long(), 'l -> b l', b=batch_size)
    # 比较生成的序列索引和每个序列的长度，生成掩码
    seq_masks = seq_masks < seq_lens.unsqueeze(1)
    # 返回生成的序列掩码
    return seq_masks

# 定义一个函数用来对张量进行填充
def pad_tensors(tensors, lens=None, pad=0, pad_ori_data=False):
    """
    参数:
    tensors: 一个张量列表，每个张量形状为[T, ...]
    lens: 可选，每个张量的长度列表
    pad: 填充值，默认为0
    pad_ori_data: 布尔值，是否在原数据后面重复填充
    返回:
    output: 填充后的张量
    """
    # 如果未提供长度，则计算每个张量的第一个维度的长度
    if lens is None:
        lens = [t.size(0) for t in tensors]
    # 计算最大长度
    max_len = max(lens)
    # 获取批次大小
    bs = len(tensors)
    # 获取其他维度的大小
    hid = list(tensors[0].size()[1:])
    # 设置输出张量的大小
    size = [bs, max_len] + hid
    # 获取数据类型
    dtype = tensors[0].dtype
    # 初始化输出张量
    output = torch.zeros(*size, dtype=dtype)
    # 使用填充值填充输出张量
    if pad:
        output.data.fill_(pad)
    # 将每个输入张量的数据复制到输出张量对应的位置
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    # 如果需要，在原数据后面重复填充数据
    if pad_ori_data:
        rt = (max_len - l) // l + 1
        for j in range(rt):
            s = l + j * l
            e = min(s + l, max_len)
            output.data[i, s: e] = t.data[:e-s]
    # 返回填充后的张量
    return output
