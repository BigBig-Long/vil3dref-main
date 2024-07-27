import os
import json
from unicodedata import bidirectional
import jsonlines
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn

# 这段代码定义了一个名为GloveGRUEncoder的PyTorch神经网络模块，它是一个基于GloVe嵌入和GRU（门控循环单元）的文本编码器。此外，代码还包含了一个预备函数prepare_glove_tokenized_data，用于准备GloVe嵌入和分词数据。
#
# 以下是GloveGRUEncoder类的主要组成部分：
#
# 初始化方法(__init__):
#
# 定义了GRU层的参数，包括隐藏层大小和层数。
# 加载预训练的GloVe嵌入向量。
# 初始化GRU层。
# 前向传播方法(forward):
#
# 使用GloVe嵌入向量将文本ID转换为嵌入表示。
# 通过GRU层处理嵌入表示，输出最后的隐藏状态。
# prepare_glove_tokenized_data函数的目的是从原始文本数据中提取词汇表，并将文本分词转换为基于GloVe嵌入的整数表示。这个函数执行以下步骤：
#
# 读取原始分词数据，并统计词汇表。
# 从预训练的GloVe模型中加载词汇的嵌入向量。
# 将词汇表中的每个词映射到一个整数ID。
# 将原始文本数据中的每个词替换为其对应的整数ID。
# 保存处理后的数据和词汇表。
# 这个文本编码器可以用于各种需要文本嵌入的下游任务，如文本分类、情感分析或文本生成。通过结合GloVe嵌入和GRU层的序列建模能力，它可以有效地捕捉文本数据中的语义和顺序信息。

class GloveGRUEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        anno_dir = '/home/shichen/scratch/datasets/referit3d/annotations/glove_tokenized'
        word_embeds = torch.from_numpy(
            np.load(os.path.join(anno_dir, 'nr3d_vocab_embeds.npy'))
        )
        self.register_buffer('word_embeds', word_embeds)
        self.gru = nn.GRU(
            input_size=300, hidden_size=hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False, dropout=0
        )
        
    def forward(self, txt_ids, txt_masks):
        txt_embeds = self.word_embeds[txt_ids]
        h_0 = torch.zeros((self.num_layers, txt_ids.size(0), self.hidden_size)).to(txt_ids.device)
        txt_embeds, _ = self.gru(txt_embeds, h_0)
        return EasyDict({
            'last_hidden_state': txt_embeds,
        })
  

def prepare_glove_tokenized_data():
    import collections

    anno_dir = '/home/shichen/scratch/datasets/referit3d/annotations'
    dataset = 'nr3d'
    outdir = os.path.join(anno_dir, 'glove_tokenized')
    os.makedirs(outdir, exist_ok=True)

    data = []
    vocab = collections.Counter()
    with jsonlines.open(os.path.join(anno_dir, 'bert_tokenized', '%s.jsonl'%dataset), 'r') as f:
        for x in f:
            data.append(x)
            for w in x['tokens']:
                vocab[w] += 1
    print(len(vocab))

    word2vec = {}
    with open('/home/shichen/scratch/datasets/pretrained/wordvecs/glove.42B.300d.txt', 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] in vocab:
                word2vec[tokens[0]] = np.array([float(x) for x in tokens[1:]], dtype=np.float32)
    
    int2word = ['unk']
    for w, c in vocab.most_common():
        if w in word2vec:
            int2word.append(w)
    print(len(int2word))
    json.dump(int2word, open(os.path.join(outdir, f'{dataset}_vocab.json'), 'w'), indent=2)

    word_embeds = [np.zeros(300, dtype=np.float32)]
    for w in int2word[1:]:
        word_embeds.append(word2vec[w])
    np.save(os.path.join(outdir, f'{dataset}_vocab_embeds.npy'), word_embeds)

    word2int = {w: i for i, w in enumerate(int2word)}
    with jsonlines.open(os.path.join(outdir, f'{dataset}.jsonl'), 'w') as outf:
        for x in data:
            x['enc_tokens'] = [word2int.get(w, 0) for w in x['tokens']]
            outf.write(x)
    

if __name__ == '__main__':
    # prepare_glove_tokenized_data()
    pass