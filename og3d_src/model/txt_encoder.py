import os  # 导入os库，用于操作系统功能，如文件路径处理
import json  # 导入json库，用于处理JSON数据
from unicodedata import bidirectional  # 导入bidirectional函数，用于处理Unicode双向字符
import jsonlines  # 导入jsonlines库，用于处理JSON行文件
import numpy as np  # 导入numpy库，用于数值计算
from easydict import EasyDict  # 导入EasyDict，提供通过属性方式访问字典值的能力
import torch  # 导入torch，PyTorch深度学习框架
import torch.nn as nn  # 导入torch.nn，用于构建网络层

# 定义GloveGRUEncoder类，继承自nn.Module，是一个基于GloVe和GRU的文本编码器
class GloveGRUEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()  # 调用父类构造函数
        self.hidden_size = hidden_size  # 设置隐藏层大小
        self.num_layers = num_layers  # 设置GRU层数
        # 设置嵌入文件的路径
        anno_dir = '/home/shichen/scratch/datasets/referit3d/annotations/glove_tokenized'
        # 加载预训练的GloVe嵌入向量
        word_embeds = torch.from_numpy(
            np.load(os.path.join(anno_dir, 'nr3d_vocab_embeds.npy'))
        )
        self.register_buffer('word_embeds', word_embeds)  # 注册buffer
        # 定义GRU层
        self.gru = nn.GRU(
            input_size=300, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=False, dropout=0
        )

    def forward(self, txt_ids, txt_masks):
        # 前向传播函数，根据文本ID获取嵌入向量
        txt_embeds = self.word_embeds[txt_ids]
        # 初始化隐状态
        h_0 = torch.zeros((self.num_layers, txt_ids.size(0), self.hidden_size)).to(txt_ids.device)
        # 通过GRU处理嵌入向量
        txt_embeds, _ = self.gru(txt_embeds, h_0)
        # 返回结果字典
        return EasyDict({
            'last_hidden_state': txt_embeds,
        })

# 准备GloVe分词数据的函数
def prepare_glove_tokenized_data():
    import collections  # 导入collections库，用于容器数据类型
    anno_dir = '/home/shichen/scratch/datasets/referit3d/annotations'  # 设置注释目录路径
    dataset = 'nr3d'  # 设置数据集名称
    outdir = os.path.join(anno_dir, 'glove_tokenized')  # 设置输出目录
    os.makedirs(outdir, exist_ok=True)  # 创建输出目录
    data = []  # 初始化数据列表
    vocab = collections.Counter()  # 创建词汇表计数器
    # 读取分词数据文件
    with jsonlines.open(os.path.join(anno_dir, 'bert_tokenized', f'{dataset}.jsonl'), 'r') as f:
        for x in f:
            data.append(x)  # 添加数据
            for w in x['tokens']:  # 遍历分词
                vocab[w] += 1  # 更新词汇表计数
    print(len(vocab))  # 打印词汇表大小
    word2vec = {}  # 初始化词向量字典
    # 读取GloVe模型文件
    with open('/home/shichen/scratch/datasets/pretrained/wordvecs/glove.42B.300d.txt', 'r') as f:
        for line in f:
            tokens = line.strip().split()  # 分割行
            if tokens[0] in vocab:  # 如果词在词汇表中
                word2vec[tokens[0]] = np.array([float(x) for x in tokens[1:]], dtype=np.float32)  # 添加词向量
    int2word = ['unk']  # 初始化词到整数映射列表
    # 为词汇表中的每个词分配一个整数ID
    for w, c in vocab.most_common():
        if w in word2vec:
            int2word.append(w)
    print(len(int2word))  # 打印整数到词的映射大小
    json.dump(int2word, open(os.path.join(outdir, f'{dataset}_vocab.json'), 'w'), indent=2)  # 保存词汇表到文件
    word_embeds = [np.zeros(300, dtype=np.float32)]  # 初始化词嵌入列表
    for w in int2word[1:]:
        word_embeds.append(word2vec[w])  # 添加词向量到列表
    np.save(os.path.join(outdir, f'{dataset}_vocab_embeds.npy'), word_embeds)  # 保存词向量到文件
    word2int = {w: i for i, w in enumerate(int2word)}  # 创建词到整数的映射
    # 保存处理后的数据到文件
    with jsonlines.open(os.path.join(outdir, f'{dataset}.jsonl'), 'w') as outf:
        for x in data:
            x['enc_tokens'] = [word2int.get(w, 0) for w in x['tokens']]  # 将分词转换为整数ID
            outf.write(x)  # 写入文件


if __name__ == '__main__':
    # prepare_glove_tokenized_data()
    pass