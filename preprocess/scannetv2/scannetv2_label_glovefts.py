import os
# 导入操作系统相关模块，用于文件和目录操作
import numpy as np
# 导入NumPy数学库，用于进行高效的数值计算
import pandas as pd
# 导入Pandas数据处理库，用于操作数据帧（DataFrame）
import json
# 导入JSON处理模块，用于读写JSON数据
import collections
# 导入collections模块，用于操作计数器等集合类型

def main():
    meta_dir = 'datasets/referit3d/annotations/meta_data'
    # 设置元数据目录的路径
    data = pd.read_csv(os.path.join(meta_dir, 'scannetv2-labels.combined.tsv'), sep='\t', header=0)
    # 读取元数据文件，文件以制表符分隔，并且有标题行
    categories = list(data['raw_category'])
    # 从数据帧中提取类别名称，并转换为列表
    print('#cat', len(categories))
    # 打印类别总数

    json.dump(
        categories,
        open(os.path.join(meta_dir, 'scannetv2_raw_categories.json'), 'w'),
        indent=2
    )
    # 将类别列表以JSON格式保存到文件，缩进为2个空格

    uniq_words = collections.Counter()
    # 初始化一个计数器对象，用于统计单词出现的次数
    for x in categories:
        for w in x.strip().split():
            uniq_words[w] += 1
    # 遍历所有类别名称，分割单词并更新计数器
    print('#uniq words', len(uniq_words))
    # 打印唯一单词的总数

    word2vec = {}
    # 初始化一个字典，用于存储单词到向量的映射
    with open('datasets/pretrained/wordvecs/glove.42B.300d.txt', 'r') as f:
    # 打开预训练的词向量文件进行读取
        for line in f:
            tokens = line.strip().split()
            # 逐行读取并按空格分割
            if tokens[0] in uniq_words:
                # 如果单词在之前的类别中出现过
                word2vec[tokens[0]] = [float(x) for x in tokens[1:]]
                # 将单词映射到其对应的向量（将字符串转换为浮点数）

    print('#word2vec', len(word2vec))
    # 打印加载到映射中的词向量数量

    cat2vec = {}
    # 初始化一个字典，用于存储类别到向量平均值的映射
    for x in categories:
        vec = []
        # 初始化一个列表，用于存储当前类别的所有单词向量
        for w in x.strip().split():
            if w in word2vec:
                # 如果单词在词向量字典中
                vec.append(word2vec[w])
            else:
                print('\t', x, w, 'no exists', uniq_words[w])
                # 如果单词不在词向量字典中，打印提示信息
        if len(vec) > 0:
            cat2vec[x] = np.mean(vec, 0).tolist()
            # 如果列表不为空，计算向量平均值，并将其转换为列表保存
        else:
            cat2vec[x] = np.zeros(300, ).tolist()
            # 如果列表为空，为当前类别创建一个零向量

        if not vec:
            print(x, 'no exists')
            # 如果类别没有对应的向量，打印提示信息

    json.dump(
        cat2vec,
        open(os.path.join(meta_dir, 'cat2glove42b.json'), 'w'),
    )
    # 将类别到向量映射以JSON格式保存到文件

if __name__ == '__main__':
    main()
# 如果此模块是作为主程序运行，则执行main函数
