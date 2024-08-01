import os
import jsonlines
import json
import numpy as np
import random
import collections
import copy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# 尝试导入本地模块，如果失败则从其他位置导入
try:
    from .common import pad_tensors, gen_seq_masks
except:
    from common import pad_tensors, gen_seq_masks

# 定义旋转角度数组
ROTATE_ANGLES = [0, np.pi / 2, np.pi, np.pi * 3 / 2]


# 定义数据集类，用于加载和处理标注数据
class GTLabelDataset(Dataset):
    def __init__(
            self, scan_id_file, anno_file, scan_dir, category_file,
            cat2vec_file=None, keep_background=False, random_rotate=False,
            max_txt_len=50, max_obj_len=80, gt_scan_dir=None, iou_replace_gt=0
    ):
        # 调用父类构造函数
        super().__init__()
        # 读取扫描ID文件，并转换为集合存储
        split_scan_ids = set([x.strip() for x in open(scan_id_file, 'r')])
        # 设置扫描数据所在目录
        self.scan_dir = scan_dir
        # 设置最大文本长度限制
        self.max_txt_len = max_txt_len
        # 设置最大对象长度限制
        self.max_obj_len = max_obj_len
        # 设置是否保留背景对象
        self.keep_background = keep_background
        # 设置是否应用随机旋转
        self.random_rotate = random_rotate
        # 设置替代的扫描数据目录（用于增强数据或使用高质量扫描）
        self.gt_scan_dir = gt_scan_dir
        # 设置交并比（IOU）门限用于替换GT（Ground Truth）数据
        self.iou_replace_gt = iou_replace_gt
        # 初始化存储扫描ID的集合
        self.scan_ids = set()
        # 初始化数据存储列表
        self.data = []
        # 初始化字典，用于存储扫描到项索引的映射
        self.scan_to_item_idxs = collections.defaultdict(list)

        # 加载注释文件
        with jsonlines.open(anno_file, 'r') as f:
            for item in f:
                # 检查注释条目的扫描ID是否在之前从文件读取的ID集合中
                if item['scan_id'] in split_scan_ids:
                    # 如果条目的文本令牌数量超过24并且item_id不是以'scanrefer'开始的，则跳过此条目
                    if (len(item['tokens']) > 24) and (not item['item_id'].startswith('scanrefer')):
                        continue
                    # 将扫描ID添加到scan_ids集合中
                    self.scan_ids.add(item['scan_id'])
                    # 将数据索引添加到scan_to_item_idxs字典中，以扫描ID为键
                    self.scan_to_item_idxs[item['scan_id']].append(len(self.data))
                    # 将注释条目添加到数据列表中
                    self.data.append(item)

        # 加载每个扫描的实例数据
        self.scans = {}
        for scan_id in self.scan_ids:
            # 读取实例标签文件
            inst_labels = json.load(open(os.path.join(scan_dir, 'instance_id_to_name', '%s.json' % scan_id)))
            # 读取实例位置文件
            inst_locs = np.load(os.path.join(scan_dir, 'instance_id_to_loc', '%s.npy' % scan_id))
            # 读取实例颜色文件
            inst_colors = json.load(open(os.path.join(scan_dir, 'instance_id_to_gmm_color', '%s.json' % scan_id)))
            # 将颜色数据转换为浮点型，并将权重和RGB值合并为一个数组
            inst_colors = [np.concatenate(
                [np.array(x['weights'])[:, None], np.array(x['means'])],
                axis=1
            ).astype(np.float32) for x in inst_colors]
            # 将处理好的实例数据（标签、位置、颜色）保存到scans字典中，以扫描ID为键
            self.scans[scan_id] = {
                'inst_labels': inst_labels,  # (n_obj, )
                'inst_locs': inst_locs,  # (n_obj, 6) 中心xyz坐标, 宽高长
                'inst_colors': inst_colors  # (n_obj, 3x4) 聚类 * (权重, 平均rgb)
            }

        # 如果定义了用于替代的扫描目录
        if self.gt_scan_dir is not None:
            for scan_id in self.scan_ids:
                # 从替代目录加载实例标签
                inst_labels = json.load(open(os.path.join(self.gt_scan_dir, 'instance_id_to_name', f'{scan_id}.json')))
                # 从替代目录加载实例位置
                inst_locs = np.load(os.path.join(self.gt_scan_dir, 'instance_id_to_loc', f'{scan_id}.npy'))
                # 从替代目录加载实例颜色
                inst_colors = json.load(
                    open(os.path.join(self.gt_scan_dir, 'instance_id_to_gmm_color', f'{scan_id}.json')))
                # 转换颜色数据格式
                inst_colors = [np.concatenate(
                    [np.array(x['weights'])[:, None], np.array(x['means'])],
                    axis=1
                ).astype(np.float32) for x in inst_colors]
                # 更新数据集中对应扫描的实例数据
                self.scans[scan_id].update({
                    'gt_inst_labels': inst_labels,  # (n_obj, )
                    'gt_inst_locs': inst_locs,  # (n_obj, 6) 中心xyz, 宽高长
                    'gt_inst_colors': inst_colors  # (n_obj, 3x4) 聚类 * (权重, 平均rgb)
                })

        # 加载类别映射文件
        self.int2cat = json.load(open(category_file, 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}

        # 加载类别到向量的映射文件（如果提供）
        if cat2vec_file is None:
            self.cat2vec = None
        else:
            self.cat2vec = json.load(open(cat2vec_file, 'r'))

    # 计算数据集大小
    def __len__(self):
        return len(self.data)

    # 数据获取函数，根据索引返回单个数据项
    def __getitem__(self, idx):
        # 从数据列表中获取指定索引的数据条目
        item = self.data[idx]
        # 获取当前条目的扫描ID
        scan_id = item['scan_id']
        # 获取目标对象的索引
        tgt_obj_idx = item['target_id']
        # 获取目标对象的类型
        tgt_obj_type = item['instance_type']

        # 处理文本数据：截取到最大文本长度的编码令牌并转换为张量
        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        # 获取文本的实际长度
        txt_lens = len(txt_tokens)

        # 根据条件判断使用原始扫描数据还是高质量的扫描数据
        if self.gt_scan_dir is None or item['max_iou'] > self.iou_replace_gt:
            # 从原始扫描数据中获取对象的标签、位置和颜色信息
            obj_labels = self.scans[scan_id]['inst_labels']
            obj_locs = self.scans[scan_id]['inst_locs']
            obj_colors = self.scans[scan_id]['inst_colors']
        else:
            # 从高质量扫描数据中获取对象的标签、位置和颜色信息
            tgt_obj_idx = item['gt_target_id']
            obj_labels = self.scans[scan_id]['gt_inst_labels']
            obj_locs = self.scans[scan_id]['gt_inst_locs']
            obj_colors = self.scans[scan_id]['gt_inst_colors']

        # 生成对象ID列表
        obj_ids = [str(x) for x in range(len(obj_labels))]

        # 如果设置为不保留背景对象
        if not self.keep_background:
            # 筛选出非背景对象的索引
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if
                                 obj_label not in ['wall', 'floor', 'ceiling']]
            # 更新目标对象索引为筛选后的索引
            tgt_obj_idx = selected_obj_idxs.index(tgt_obj_idx)
            # 更新对象的标签、位置和颜色信息为非背景对象的数据
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_locs = obj_locs[selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]

        # 如果设置了随机旋转
        if self.random_rotate:
            # 随机选择一个旋转角度
            theta_idx = np.random.randint(len(ROTATE_ANGLES))
            theta = ROTATE_ANGLES[theta_idx]
        else:
            # 不进行旋转
            theta = 0
        # 获取增强后的对象输入
        aug_obj_labels, aug_obj_locs, aug_obj_colors, aug_obj_ids, aug_tgt_obj_idx = \
            self._get_obj_inputs(
                obj_labels, obj_locs, obj_colors, obj_ids, tgt_obj_idx,
                theta=theta
            )
        # 将增强后的对象位置和颜色转换为张量
        aug_obj_locs = torch.from_numpy(aug_obj_locs)
        aug_obj_colors = torch.from_numpy(aug_obj_colors)
        # 将对象类别标签转换为类别索引的张量
        aug_obj_classes = torch.LongTensor([self.cat2int[x] for x in aug_obj_labels])

        # 根据是否有类别到向量的映射，选择对象特征的表示方式
        if self.cat2vec is None:
            # 如果没有提供类别到向量的映射，直接使用类别索引
            aug_obj_fts = aug_obj_classes
        else:
            # 如果提供了映射，则使用向量表示对象特征
            aug_obj_fts = torch.FloatTensor([self.cat2vec[x] for x in aug_obj_labels])

        # 组装最终输出的字典
        outs = {
            'item_ids': item['item_id'],  # 数据条目的ID
            'scan_ids': scan_id,  # 扫描ID
            'txt_ids': txt_tokens,  # 文本数据的张量
            'txt_lens': txt_lens,  # 文本长度
            'obj_fts': aug_obj_fts,  # 对象特征
            'obj_locs': aug_obj_locs,  # 对象位置
            'obj_colors': aug_obj_colors,  # 对象颜色
            'obj_lens': len(aug_obj_fts),  # 对象数量
            'obj_classes': aug_obj_classes,  # 对象类别
            'tgt_obj_idxs': aug_tgt_obj_idx,  # 目标对象的索引
            'tgt_obj_classes': self.cat2int[tgt_obj_type],  # 目标对象的类别索引
            'obj_ids': aug_obj_ids,  # 对象的ID列表
        }
        return outs

    # 自定义集合函数，用于组织批次数据
    def gtlabel_collate_fn(data):
        outs = {}
        # 遍历批次中第一个样本的所有键，以初始化输出字典
        for key in data[0].keys():
            # 收集同一键下所有样本的值
            outs[key] = [x[key] for x in data]

        # 对文本ID进行填充以处理不同长度的序列
        outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
        # 转换文本长度列表为张量
        outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
        # 生成文本序列的掩码，用于训练时忽略填充部分
        outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])

        # 对象特征可能是一维（如果它们是索引）或多维（如果它们是向量），这决定了填充方式
        if len(outs['obj_fts'][0].size()) == 1:
            # 如果是一维，直接填充序列
            outs['obj_fts'] = pad_sequence(outs['obj_fts'], batch_first=True)
        else:
            # 如果是多维，使用自定义填充函数处理不同长度的张量
            outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'])

        # 填充对象位置和颜色张量
        outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0)
        outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0)

        # 转换对象长度列表为张量
        outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
        # 生成对象序列的掩码
        outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])

        # 对象类别可能需要特殊值填充（如-100），通常用于忽略某些类别计算损失
        outs['obj_classes'] = pad_sequence(
            outs['obj_classes'], batch_first=True, padding_value=-100
        )

        # 目标对象索引和类别转换为张量
        outs['tgt_obj_idxs'] = torch.LongTensor(outs['tgt_obj_idxs'])
        outs['tgt_obj_classes'] = torch.LongTensor(outs['tgt_obj_classes'])

        # 返回整理好的批次数据
        return outs

