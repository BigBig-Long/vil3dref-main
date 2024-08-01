import os
# 导入os模块，用于文件和目录操作
import argparse
# 导入argparse模块，用于解析命令行参数
import json
# 导入json模块，用于处理JSON数据
import jsonlines
# 导入jsonlines模块，用于读写JSON Lines格式文件
import pandas as pd
# 导入pandas模块，用于数据处理
from transformers import AutoTokenizer, AutoModel
# 从transformers库导入AutoTokenizer和AutoModel，用于处理BERT模型

def main():
    parser = argparse.ArgumentParser()
    # 创建一个ArgumentParser对象，用于解析命令行参数
    parser.add_argument('--input_file')
    # 添加一个命令行参数，用于指定输入文件
    parser.add_argument('--output_file')
    # 添加一个命令行参数，用于指定输出文件
    args = parser.parse_args()
    # 解析命令行参数
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # 确保输出文件所在的目录存在，如果不存在则创建

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # 从预训练的BERT模型加载分词器

    data = pd.read_csv(args.input_file, header=0)
    # 读取输入的CSV文件，假设第一行是标题行

    print('process %d items' % (len(data)))
    # 打印正在处理的数据项数量

    num_reserved_items = 0
    # 初始化保留的数据项数量计数器

    with jsonlines.open(args.output_file, 'w') as outf:
        # 打开输出文件，准备写入JSON Lines格式的数据
        for i in range(len(data)):
            item = data.iloc[i]
            # 获取数据帧中的当前数据项

            if not item['mentions_target_class']:
                continue
            # 如果当前数据项没有提到目标类，则跳过

            enc_tokens = tokenizer.encode(item['utterance'])
            # 使用分词器对当前数据项的“utterance”字段进行编码

            new_item = {
                # 构建一个新的字典，包含处理后的数据项
                'item_id': '%s_%06d' % (item['dataset'], i),
                # 创建一个唯一的数据项ID
                'stimulus_id': item['stimulus_id'],
                # 添加刺激ID
                'scan_id': item['scan_id'],
                # 添加扫描ID
                'instance_type': item['instance_type'],
                # 添加实例类型
                'target_id': int(item['target_id']),
                # 添加目标ID（转换为整数）
                'utterance': item['utterance'],
                # 添加原始语句
                'tokens': eval(item['tokens']),
                # 将字符串转换为列表（假设'tokens'字段是字符串表示的列表）
                'enc_tokens': enc_tokens,
                # 添加编码后的标记
                'correct_guess': bool(item['correct_guess']),
                # 添加正确猜测的布尔值
            }

            if item['dataset'] == 'nr3d':
                # 如果数据集是nr3d，则添加额外的字段
                new_item.update({
                    'uses_object_lang': bool(item['uses_object_lang']),
                    'uses_spatial_lang': bool(item['uses_spatial_lang']),
                    'uses_color_lang': bool(item['uses_color_lang']),
                    'uses_shape_lang': bool(item['uses_shape_lang'])
                })
            else:
                # 否则，添加其他数据集的字段
                new_item.update({
                    'coarse_reference_type': item['coarse_reference_type'],
                    'reference_type': item['reference_type'],
                    'anchors_types': eval(item['anchors_types']),
                    'anchor_ids': eval(item['anchor_ids']),
                })

            # for k, v in new_item.items():
            #     print(k, type(v))
            # 注释掉的代码用于打印新数据项的键和值类型

            outf.write(new_item)
            # 将处理后的数据项写入输出文件

            num_reserved_items += 1
            # 更新保留的数据项数量

    print('keep %d items' % (num_reserved_items))
    # 打印保留的数据项数量

if __name__ == '__main__':
    main()
# 如果此模块是作为主程序运行，则执行main函数
