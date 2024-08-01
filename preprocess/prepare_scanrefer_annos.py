import os
# 导入os模块，用于文件和目录操作
import argparse
# 导入argparse模块，用于解析命令行参数
import json
# 导入json模块，用于处理JSON数据
import jsonlines
# 导入jsonlines模块，用于读写JSON Lines格式文件
from transformers import AutoTokenizer, AutoModel
# 从transformers库导入AutoTokenizer和AutoModel，用于处理BERT模型

def main():
    parser = argparse.ArgumentParser()
    # 创建一个ArgumentParser对象，用于解析命令行参数
    parser.add_argument('--input_dir')
    # 添加一个命令行参数，用于指定输入目录
    parser.add_argument('--output_file')
    # 添加一个命令行参数，用于指定输出文件
    args = parser.parse_args()
    # 解析命令行参数

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # 确保输出文件所在的目录存在，如果不存在则创建

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # 从预训练的BERT模型加载分词器

    num_reserved_items = 0
    # 初始化保留的数据项数量计数器

    with jsonlines.open(args.output_file, 'w') as outf:
        # 打开输出文件，准备写入JSON Lines格式的数据
        for split in ['train', 'val', 'test']:
            # 遍历数据集的三个分割：训练集、验证集和测试集
            data = json.load(open(os.path.join(args.input_dir, 'ScanRefer_filtered_%s.json'%split)))
            # 加载对应分割的JSON文件数据

            print('process %s: %d items' % (split, len(data)))
            # 打印当前分割的数据项数量

            for i, item in enumerate(data):
                # 遍历数据集中的每个项目
                enc_tokens = tokenizer.encode(item['description'])
                # 使用分词器对描述进行编码

                # 构建新的数据项字典，并写入输出文件
                outf.write({
                    'item_id': 'scanrefer_%s_%06d' % (split, i),
                    # 创建唯一的数据项ID
                    'scan_id': item['scene_id'],
                    # 添加场景ID
                    'target_id': int(item['object_id']),
                    # 添加目标ID（转换为整数）
                    'instance_type': item['object_name'].replace('_', ' '),
                    # 添加实例类型，并将下划线替换为空格
                    'utterance': item['description'],
                    # 添加描述
                    'tokens': item['token'],
                    # 添加分词结果
                    'enc_tokens': enc_tokens,
                    # 添加编码后的标记
                    'ann_id': item['ann_id']
                    # 添加注释ID
                })
                num_reserved_items += 1
                # 更新保留的数据项数量

    print('keep %d items' % (num_reserved_items))
    # 打印保留的数据项总数

if __name__ == '__main__':
    main()
# 如果此模块是作为主程序运行，则执行main函数
