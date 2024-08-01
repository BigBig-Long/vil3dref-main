# 导入必要的Python模块
import os
import jsonlines
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
# 尝试从当前目录下的common模块中导入函数，如果失败，则从上级目录导入
try:
    from .common import pad_tensors, gen_seq_masks
    from .gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES
except:
    from common import pad_tensors, gen_seq_masks
    from gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES

# 定义GTLabelPcdDataset类，继承自GTLabelDataset
class GTLabelPcdDataset(GTLabelDataset):
    def __init__(
        self, scan_id_file, anno_file, scan_dir, category_file,
        cat2vec_file=None, keep_background=False, random_rotate=False,
        num_points=1024, max_txt_len=50, max_obj_len=80,
        in_memory=False, gt_scan_dir=None, iou_replace_gt=0
    ):
        # 调用父类的初始化方法
        super().__init__(
            scan_id_file, anno_file, scan_dir, category_file,
            cat2vec_file=cat2vec_file, keep_background=keep_background,
            random_rotate=random_rotate,
            max_txt_len=max_txt_len, max_obj_len=max_obj_len,
            gt_scan_dir=gt_scan_dir, iou_replace_gt=iou_replace_gt
        )
        # 设置点云数据中点的数量
        self.num_points = num_points
        # 设置是否将数据加载到内存中
        self.in_memory = in_memory
        # 如果设置将数据加载到内存，则预处理所有扫描数据的点云
        if self.in_memory:
            for scan_id in self.scan_ids:
                self.get_scan_pcd_data(scan_id)

    # 获取点云数据的方法
    def get_scan_pcd_data(self, scan_id):
        # 如果数据已经在内存中，则直接返回
        if self.in_memory and 'pcds' in self.scans[scan_id]:
            return self.scans[scan_id]['pcds']
        # 加载点云数据
        pcd_data = torch.load(
            os.path.join(self.scan_dir, 'pcd_with_global_alignment', '%s.pth'%scan_id)
        )
        points, colors = pcd_data[0], pcd_data[1]
        # 归一化颜色数据
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)
        instance_labels = pcd_data[-1]
        # 根据实例标签分割点云
        obj_pcds = []
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i     # 这个操作可能比较耗时
            obj_pcds.append(pcds[mask])
        # 如果设置将数据加载到内存，则保存分割后的点云数据
        if self.in_memory:
            self.scans[scan_id]['pcds'] = obj_pcds
        return obj_pcds

    # 获取真实点云数据的方法（类似于上一个方法，但是从不同的目录加载）
    def get_scan_gt_pcd_data(self, scan_id):
        # 如果数据已经在内存中，则直接返回
        if self.in_memory and 'gt_pcds' in self.scans[scan_id]:
            return self.scans[scan_id]['gt_pcds']
        # 加载真实点云数据
        pcd_data = torch.load(
            os.path.join(self.gt_scan_dir, 'pcd_with_global_alignment', '%s.pth'%scan_id)
        )
        points, colors = pcd_data[0], pcd_data[1]
        # 归一化颜色数据
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)
        instance_labels = pcd_data[-1]
        # 根据实例标签分割点云
        obj_pcds = []
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i     # 这个操作可能比较耗时
            obj_pcds.append(pcds[mask])
        # 如果设置将数据加载到内存，则保存分割后的点云数据
        if self.in_memory:
            self.scans[scan_id]['gt_pcds'] = obj_pcds
        return obj_pcds

    # 获取对象输入数据的方法
    def _get_obj_inputs(self, obj_pcds, obj_colors, obj_labels, obj_ids, tgt_obj_idx, theta=None):
        # 获取目标对象的类型
        tgt_obj_type = obj_labels[tgt_obj_idx]
        # 如果设置了最大对象长度，并且实际对象数量超过这个长度，则进行筛选
        if (self.max_obj_len is not None) and (len(obj_labels) > self.max_obj_len):
            selected_obj_idxs = [tgt_obj_idx]
            remained_obj_idxs = []
            # 遍历所有对象，根据类型进行筛选
            for kobj, klabel in enumerate(obj_labels):
                if kobj != tgt_obj_idx:
                    if klabel == tgt_obj_type:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idxs.append(kobj)
            # 如果筛选后的对象数量少于最大长度，则随机补充剩余的对象
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idxs)
                selected_obj_idxs += remained_obj_idxs[:self.max_obj_len - len(selected_obj_idxs)]
            # 根据筛选后的索引更新对象数据
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]
            tgt_obj_idx = 0  # 重置目标对象索引为0

        # 如果需要旋转，则生成旋转矩阵
        if (theta is not None) and (theta != 0):
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            rot_matrix = None

        obj_fts, obj_locs = [], []
        for obj_pcd in obj_pcds:
            # 如果有旋转矩阵，则对点云数据进行旋转
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            # 计算对象的中心点和大小
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            # 采样点云数据
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
            obj_pcd = obj_pcd[pcd_idxs]
            # 归一化点云数据
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3] ** 2, 1)))
            if max_dist < 1e-6:  # 处理非常小的点云，即填充
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        obj_fts = np.stack(obj_fts, 0)
        obj_locs = np.array(obj_locs)
        obj_colors = np.array(obj_colors)
        return obj_fts, obj_locs, obj_colors, obj_labels, obj_ids, tgt_obj_idx

        # 重写Dataset类的__getitem__方法，用于获取单个数据项

    def __getitem__(self, idx):
        # 获取数据项
        item = self.data[idx]
        # 获取扫描ID
        scan_id = item['scan_id']
        # 获取文本令牌
        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        # 获取目标对象索引和类型
        tgt_obj_idx = item['target_id']
        tgt_obj_type = item['instance_type']

        

        # 根据是否有gt_scan_dir和iou_replace_gt的设置，获取点云数据
        if self.gt_scan_dir is None or item['max_iou'] > self.iou_replace_gt:
            obj_pcds = self.get_scan_pcd_data(scan_id)
            obj_labels = self.scans[scan_id]['inst_labels']
            obj_gmm_colors = self.scans[scan_id]['inst_colors']
        else:
            tgt_obj_idx = item['gt_target_id']
            obj_pcds = self.get_scan_gt_pcd_data(scan_id)
            obj_labels = self.scans[scan_id]['gt_inst_labels']
            obj_gmm_colors = self.scans[scan_id]['gt_inst_colors']
        obj_ids = [str(x) for x in range(len(obj_labels))]
        # 如果不需要保留背景，则进行
        # 如果不需要保留背景，则进行筛选
        if not self.keep_background:
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if obj_label not in ['wall', 'floor', 'ceiling']]
            tgt_obj_idx = selected_obj_idxs.index(tgt_obj_idx)
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_gmm_colors = [obj_gmm_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]

        # 如果需要随机旋转，则生成旋转角度
        if self.random_rotate:
            theta_idx = np.random.randint(len(ROTATE_ANGLES))
            theta = ROTATE_ANGLES[theta_idx]
        else:
            theta = 0

        # 获取增强后的对象特征、位置、颜色、标签和ID
        aug_obj_fts, aug_obj_locs, aug_obj_gmm_colors, aug_obj_labels, aug_obj_ids, aug_tgt_obj_idx = self._get_obj_inputs(
            obj_pcds, obj_gmm_colors, obj_labels, obj_ids, tgt_obj_idx, theta=theta
        )

        # 将numpy数组转换为torch张量
        aug_obj_fts = torch.from_numpy(aug_obj_fts)
        aug_obj_locs = torch.from_numpy(aug_obj_locs)
        aug_obj_gmm_colors = torch.from_numpy(aug_obj_gmm_colors)
        aug_obj_classes = torch.LongTensor([self.cat2int[x] for x in aug_obj_labels])

        # 如果有类别到向量的映射，则使用向量表示，否则使用类别ID
        if self.cat2vec is None:
            aug_obj_gt_fts = aug_obj_classes
        else:
            aug_obj_gt_fts = torch.FloatTensor([self.cat2vec[x] for x in aug_obj_labels])

        # 组装输出数据
        outs = {
            'item_ids': item['item_id'],
            'scan_ids': scan_id,
            'txt_ids': txt_tokens,
            'txt_lens': len(txt_tokens),
            'obj_gt_fts': aug_obj_gt_fts,
            'obj_fts': aug_obj_fts,
            'obj_locs': aug_obj_locs,
            'obj_colors': aug_obj_gmm_colors,
            'obj_lens': len(aug_obj_fts),
            'obj_classes': aug_obj_classes,
            'tgt_obj_idxs': aug_tgt_obj_idx,
            'tgt_obj_classes': self.cat2int[tgt_obj_type],
            'obj_ids': aug_obj_ids,
        }
        return outs

# 定义数据批处理函数
# 定义数据批处理函数
def gtlabelpcd_collate_fn(data):
    # 创建一个字典用于存储批处理后的数据
    outs = {}
    # 遍历数据中的所有键
    for key in data[0].keys():
        # 将每个键对应的数据列表存储到outs字典中
        outs[key] = [x[key] for x in data]
    # 对文本ID进行填充，以确保批次中的文本长度一致
    outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
    # 将文本长度转换为torch.LongTensor
    outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
    # 生成文本序列的掩码
    outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])
    # 对对象的真实特征进行填充
    outs['obj_gt_fts'] = pad_tensors(outs['obj_gt_fts'], lens=outs['obj_lens'])
    # 对对象的点云特征进行填充，并保留原始数据
    outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'], pad_ori_data=True)
    # 对对象的位置进行填充
    outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0)
    # 对对象的颜色进行填充
    outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0)
    # 将对象长度转换为torch.LongTensor
    outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
    # 生成对象序列的掩码
    outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])
    # 对对象的类别进行填充，填充值为-100
    outs['obj_classes'] = pad_sequence(
        outs['obj_classes'], batch_first=True, padding_value=-100
    )
    # 将目标对象索引转换为torch.LongTensor
    outs['tgt_obj_idxs'] = torch.LongTensor(outs['tgt_obj_idxs'])
    # 将目标对象类别转换为torch.LongTensor
    outs['tgt_obj_classes'] = torch.LongTensor(outs['tgt_obj_classes'])
    # 返回批处理后的数据
    return outs

