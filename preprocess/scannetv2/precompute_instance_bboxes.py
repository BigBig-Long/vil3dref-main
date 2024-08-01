

import os
import numpy as np
import torch
from tqdm import tqdm
import argparse


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加两个必须的参数：点云数据目录和边界框输出目录
    parser.add_argument('pcd_data_dir')
    parser.add_argument('bbox_out_dir')
    # 解析命令行参数
    args = parser.parse_args()

    # 创建输出目录，如果目录已存在则忽略错误
    os.makedirs(args.bbox_out_dir, exist_ok=True)

    # 获取点云数据目录下的所有文件名，并提取出scan_id
    scan_ids = [x.split('.')[0] for x in os.listdir(args.pcd_data_dir)]
    # 对scan_id进行排序
    scan_ids.sort()

    # 遍历所有scan_id
    for scan_id in tqdm(scan_ids):
        # 加载点云数据，包括点、颜色、标签和实例标签
        points, colors, _, inst_labels = torch.load(
            os.path.join(args.pcd_data_dir, '%s.pth' % scan_id)
        )
        # 如果没有实例标签，则跳过当前scan_id
        if inst_labels is None:
            continue

        # 获取实例的总数（最大实例标签）
        num_insts = inst_labels.max()
        # 初始化输出列表
        outs = []

        # 遍历每个实例
        for i in range(num_insts + 1):
            # 获取当前实例的掩码
            inst_mask = inst_labels == i
            # 获取当前实例的点云数据
            inst_points = points[inst_mask]

            # 如果当前实例的点云数据为空，则输出提示信息并添加一个全0的边界框
            if len(inst_points) == 0:
                print(scan_id, i, 'empty bbox')
                outs.append(np.zeros(6, ).astype(np.float32))
            else:
                # 计算边界框的中心和大小
                bbox_center = inst_points.mean(0)
                bbox_size = inst_points.max(0) - inst_points.min(0)
                # 将中心和大小的数据拼接并添加到输出列表
                outs.append(np.concatenate([bbox_center, bbox_size], 0))

        # 将输出列表转换为numpy数组并保存为.npy文件
        outs = np.stack(outs, 0).astype(np.float32)
        np.save(os.path.join(args.bbox_out_dir, '%s.npy' % scan_id), outs)


# 当该脚本作为主程序运行时，调用main函数
if __name__ == '__main__':
    main()
