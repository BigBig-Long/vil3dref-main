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
        super().__init__()
        # 从文件读取扫描ID并设置相关变量
        split_scan_ids = set([x.strip() for x in open(scan_id_file, 'r')])
        self.scan_dir = scan_dir
        self.max_txt_len = max_txt_len
        self.max_obj_len = max_obj_len
        self.keep_background = keep_background
        self.random_rotate = random_rotate
        self.gt_scan_dir = gt_scan_dir
        self.iou_replace_gt = iou_replace_gt
        self.scan_ids = set()
        self.data = []
        self.scan_to_item_idxs = collections.defaultdict(list)

        # 加载注释文件
        with jsonlines.open(anno_file, 'r') as f:
            for item in f:
                if item['scan_id'] in split_scan_ids:
                    if (len(item['tokens']) > 24) and (not item['item_id'].startswith('scanrefer')):
                        continue
                    # 根据注释创建内部数据结构
                    self.scan_ids.add(item['scan_id'])
                    self.scan_to_item_idxs[item['scan_id']].append(len(self.data))
                    self.data.append(item)

        # 加载扫描数据
        self.scans = {}
        for scan_id in self.scan_ids:
            inst_labels = json.load(open(os.path.join(scan_dir, 'instance_id_to_name', '%s.json' % scan_id)))
            inst_locs = np.load(os.path.join(scan_dir, 'instance_id_to_loc', '%s.npy' % scan_id))
            inst_colors = json.load(open(os.path.join(scan_dir, 'instance_id_to_gmm_color', '%s.json' % scan_id)))
            inst_colors = [np.concatenate(
                [np.array(x['weights'])[:, None], np.array(x['means'])],
                axis=1
            ).astype(np.float32) for x in inst_colors]
            self.scans[scan_id] = {
                'inst_labels': inst_labels,  # (n_obj, )
                'inst_locs': inst_locs,  # (n_obj, 6) center xyz, whl
                'inst_colors': inst_colors,  # (n_obj, 3x4) cluster * (weight, mean rgb)
            }
        # 如果定义了用于替代的扫描目录
        if self.gt_scan_dir is not None:
            for scan_id in self.scan_ids:
                inst_labels = json.load(open(os.path.join(gt_scan_dir, 'instance_id_to_name', '%s.json' % scan_id)))
                inst_locs = np.load(os.path.join(gt_scan_dir, 'instance_id_to_loc', '%s.npy' % scan_id))
                inst_colors = json.load(
                    open(os.path.join(gt_scan_dir, 'instance_id_to_gmm_color', '%s.json' % scan_id)))
                inst_colors = [np.concatenate(
                    [np.array(x['weights'])[:, None], np.array(x['means'])],
                    axis=1
                ).astype(np.float32) for x in inst_colors]
                self.scans[scan_id].update({
                    'gt_inst_labels': inst_labels,  # (n_obj, )
                    'gt_inst_locs': inst_locs,  # (n_obj, 6) center xyz, whl
                    'gt_inst_colors': inst_colors,  # (n_obj, 3x4) cluster * (weight, mean rgb)
                })
        # 加载类别映射文件
        self.int2cat = json.load(open(category_file, 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        if cat2vec_file is None:
            self.cat2vec = None
        else:
            self.cat2vec = json.load(open(cat2vec_file, 'r'))

    # 计算数据集大小
    def __len__(self):
        return len(self.data)

    # 数据获取函数，根据索引返回单个数据项
    def __getitem__(self, idx):
        item = self.data[idx]
        scan_id = item['scan_id']
        tgt_obj_idx = item['target_id']
        tgt_obj_type = item['instance_type']
        # 处理文本数据
        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        txt_lens = len(txt_tokens)
        # 处理对象数据
        if self.gt_scan_dir is None or item['max_iou'] > self.iou_replace_gt:
            obj_labels = self.scans[scan_id]['inst_labels']
            obj_locs = self.scans[scan_id]['inst_locs']
            obj_colors = self.scans[scan_id]['inst_colors']
        else:
            tgt_obj_idx = item['gt_target_id']
            obj_labels = self.scans[scan_id]['gt_inst_labels']
            obj_locs = self.scans[scan_id]['gt_inst_locs']
            obj_colors = self.scans[scan_id]['gt_inst_colors']
        obj_ids = [str(x) for x in range(len(obj_labels))]
        if not self.keep_background:
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if
                                 obj_label not in ['wall', 'floor', 'ceiling']]
            tgt_obj_idx = selected_obj_idxs.index(tgt_obj_idx)
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_locs = obj_locs[selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]
        if self.random_rotate:
            theta_idx = np.random.randint(len(ROTATE_ANGLES))
            theta = ROTATE_ANGLES[theta_idx]
        else:
            theta = 0
        # 获取增强后的对象输入
        aug_obj_labels, aug_obj_locs, aug_obj_colors, aug_obj_ids, aug_tgt_obj_idx = \
            self._get_obj_inputs(
                obj_labels, obj_locs, obj_colors, obj_ids, tgt_obj_idx,
                theta=theta
            )
        aug_obj_locs = torch.from_numpy(aug_obj_locs)
        aug_obj_colors = torch.from_numpy(aug_obj_colors)
        aug_obj_classes = torch.LongTensor([self.cat2int[x] for x in aug_obj_labels])
        if self.cat2vec is None:
            aug_obj_fts = aug_obj_classes
        else:
            aug_obj_fts = torch.FloatTensor([self.cat2vec[x] for x in aug_obj_labels])
        # 返回处理后的数据
        outs = {
            'item_ids': item['item_id'],
            'scan_ids': scan_id,
            'txt_ids': txt_tokens,
            'txt_lens': txt_lens,
            'obj_fts': aug_obj_fts,
            'obj_locs': aug_obj_locs,
            'obj_colors': aug_obj_colors,
            'obj_lens': len(aug_obj_fts),
            'obj_classes': aug_obj_classes,
            'tgt_obj_idxs': aug_tgt_obj_idx,
            'tgt_obj_classes': self.cat2int[tgt_obj_type],
            'obj_ids': aug_obj_ids,
        }
        return outs

    # 自定义集合函数，用于组织批次数据
    def gtlabel_collate_fn(data):
        outs = {}
        for key in data[0].keys():
            outs[key] = [x[key] for x in data]
        outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
        outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
        outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])
        if len(outs['obj_fts'][0].size()) == 1:
            outs['obj_fts'] = pad_sequence(outs['obj_fts'], batch_first=True)
        else:
            outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'])
        outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0)
        outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0)
        outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
        outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])
        outs['obj_classes'] = pad_sequence(
            outs['obj_classes'], batch_first=True, padding_value=-100
        )
        outs['tgt_obj_idxs'] = torch.LongTensor(outs['tgt_obj_idxs'])
        outs['tgt_obj_classes'] = torch.LongTensor(outs['tgt_obj_classes'])
        return outs
