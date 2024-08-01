import os
import argparse
import json
import numpy as np
import pprint
import time
import multiprocessing as mp
from functools import partial

from plyfile import PlyData

import torch


# 定义处理单个扫描数据的函数
def process_per_scan(scan_id, scan_dir, out_dir, apply_global_alignment=True, is_test=False):
    # 根据是否应用全局对齐来决定输出PCD文件的目录
    pcd_out_dir = os.path.join(out_dir,
                               'pcd_with_global_alignment' if apply_global_alignment else 'pcd_no_global_alignment')
    os.makedirs(pcd_out_dir, exist_ok=True)  # 创建目录，如果已存在则不报错

    # 准备输出对象ID到名称的目录
    obj_out_dir = os.path.join(out_dir, 'instance_id_to_name')
    os.makedirs(obj_out_dir, exist_ok=True)  # 创建目录，如果已存在则不报错

    # 加载带颜色的点云数据
    with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.ply' % (scan_id)), 'rb') as f:
        plydata = PlyData.read(f)  # 读取PLY数据，包含顶点（vertex）和面（face）信息
        points = np.array([list(x) for x in plydata.elements[0]])  # 获取点数据，格式为[[x, y, z, r, g, b, alpha]]
        coords = np.ascontiguousarray(points[:, :3])  # 获取坐标信息
        colors = np.ascontiguousarray(points[:, 3:6])  # 获取颜色信息

    # # TODO: 对坐标和颜色进行归一化（这段代码被注释掉了）
    # coords = coords - coords.mean(0)  # 对坐标进行中心化
    # colors = colors / 127.5 - 1  # 对颜色进行归一化

    # 如果应用全局对齐
    if apply_global_alignment:
        align_matrix = np.eye(4)  # 初始化4x4单位矩阵
        # 读取全局对齐矩阵
        with open(os.path.join(scan_dir, scan_id, '%s.txt' % (scan_id)), 'r') as f:
            for line in f:
                if line.startswith('axisAlignment'):
                    align_matrix = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(
                        4, 4)
                    break  # 找到对齐矩阵后跳出循环

    # 转换点云坐标
    pts = np.ones((coords.shape[0], 4), dtype=coords.dtype)  # 创建一个N*4的矩阵，最后一列全为1（齐次坐标）
    pts[:, 0:3] = coords  # 将坐标信息复制到前3列
    coords = np.dot(pts, align_matrix.transpose())[:, :3]  # 应用对齐矩阵进行坐标变换

    # 确保转换后没有引入NAN值
    assert (np.sum(np.isnan(coords)) == 0)  # 如果有NAN值，抛出断言错误
    # 如果是测试模式，则不加载标签
    if is_test:
        sem_labels = None  # 语义标签设置为空
        instance_labels = None  # 实例标签设置为空
    else:
        # 如果不是测试模式，则从文件中读取标签
        # 打开包含NYU40标签的PLY文件（每个点的类别标签）
        with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.labels.ply' % (scan_id)), 'rb') as f:
            plydata = PlyData.read(f)  # 读取PLY数据
            sem_labels = np.array(plydata.elements[0]['label']).astype(np.long)  # 读取标签数据，转换为长整型

        # 确保点坐标、颜色和语义标签的数量相同
        assert len(coords) == len(colors) == len(sem_labels)

        # 读取将每个点映射到片段ID的文件
        with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json' % (scan_id)), 'r') as f:
            d = json.load(f)  # 加载JSON数据
            seg = d['segIndices']  # 获取片段索引

        # 构建一个从片段ID到点ID的映射
        segid_to_pointid = {}
        for i, segid in enumerate(seg):
            segid_to_pointid.setdefault(segid, []).append(i)  # 为每个片段ID添加对应的点ID列表

        # 读取将对象映射到片段的文件
        instance_class_labels = []  # 实例类别标签列表
        instance_segids = []  # 实例片段ID列表
        with open(os.path.join(scan_dir, scan_id, '%s.aggregation.json' % (scan_id)), 'r') as f:
            d = json.load(f)  # 加载JSON数据
            for i, x in enumerate(d['segGroups']):
                # 确保组ID、对象ID和索引是一致的
                assert x['id'] == x['objectId'] == i
                instance_class_labels.append(x['label'])  # 添加实例类别标签
                instance_segids.append(x['segments'])  # 添加包含此实例的片段ID列表

        # 初始化实例标签数组，默认为-100（表示未赋值）
        instance_labels = np.ones(sem_labels.shape[0], dtype=np.long) * -100

        # 遍历所有实例及其片段，为每个点分配实例标签
        for i, segids in enumerate(instance_segids):
            pointids = []  # 存储属于当前实例的点ID
            for segid in segids:
                pointids += segid_to_pointid[segid]  # 添加此片段的所有点ID
            # 如果这些点已经有标签，则打印警告信息（可能存在重叠实例）
            if np.sum(instance_labels[pointids] != -100) > 0:
                print(scan_id, i, np.sum(instance_labels[pointids] != -100), len(pointids))
            else:
                instance_labels[pointids] = i  # 为这些点分配当前实例的标签

        # 确保每个实例中的点具有相同的标签
        assert len(np.unique(sem_labels[pointids])) == 1, 'points of each instance should have the same label'

        # 将实例类别标签保存到JSON文件
        json.dump(
            instance_class_labels,
            open(os.path.join(obj_out_dir, '%s.json' % scan_id), 'w'),
            indent=2
        )

        # 将坐标、颜色、语义标签和实例标签保存到PTH文件
        torch.save(
            (coords, colors, sem_labels, instance_labels),
            os.path.join(pcd_out_dir, '%s.pth' % (scan_id))
        )

import argparse
from pprint import pprint

def parse_args():
    # 创建一个 ArgumentParser 对象，它将自动生成帮助和用法提示
    parser = argparse.ArgumentParser()

    # 添加必须的参数
    parser.add_argument(
        '--scannet_dir',  # 参数名称
        required=True,    # 此参数为必须提供
        type=str,         # 参数类型为字符串
        help='the path to the downloaded ScanNet scans'  # 参数帮助信息
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        type=str,
        help='the path of the directory to be saved preprocessed scans'
    )

    # 添加可选参数
    parser.add_argument(
        '--num_workers',
        default=-1,       # 如果用户没有提供此参数，则默认值为-1
        type=int,         # 参数类型为整数
        help='the number of processes, -1 means use the available max'
    )
    parser.add_argument(
        '--apply_global_alignment',
        default=False,    # 如果用户没有提供此参数，则默认值为False
        action='store_true',  # 这是一个开关，提供此参数时，该值为True
        help='rotate/translate entire scan globally to aligned it with other scans'
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 使用 pprint 模块以更可读的格式打印参数
    args_string = pprint.pformat(vars(args))
    print(args_string)

    # 返回解析后的参数对象
    return args


def main():
    # 解析命令行传入的参数
    args = parse_args()

    # 如果输出目录不存在，则创建该目录；exist_ok=True表示如果目录已存在不会抛出异常
    os.makedirs(args.output_dir, exist_ok=True)

    # 遍历数据集的划分，这里只处理'scans'，注释掉了'scans_test'
    for split in ['scans']:

        # 拼接ScanNet数据集目录和当前划分的路径
        scannet_dir = os.path.join(args.scannet_dir, split)

        # 定义一个偏函数fn，固定某些参数，简化后续调用
        # process_per_scan是一个处理单个scan的函数
        fn = partial(
            process_per_scan,
            scan_dir=scannet_dir,
            out_dir=args.output_dir,
            apply_global_alignment=args.apply_global_alignment,
            is_test='test' in split  # 如果划分名称包含'test'，则is_test为True
        )

        # 获取ScanNet目录下的所有scan的ID
        scan_ids = os.listdir(scannet_dir)

        # 对scan ID进行排序，确保处理顺序
        scan_ids.sort()

        # 打印当前划分名称和scan的数量
        print(split, '%d scans' % (len(scan_ids)))

        # 记录开始处理的时间
        start_time = time.time()

        # 如果用户没有指定工作进程数（即args.num_workers == -1），则设置为CPU核心数和scan数量的最小值
        if args.num_workers == -1:
            num_workers = min(mp.cpu_count(), len(scan_ids))

        # 创建一个进程池，最大进程数为num_workers
        pool = mp.Pool(num_workers)

        # 使用进程池中的多个进程对每个scan并行执行fn函数
        pool.map(fn, scan_ids)

        # 关闭进程池，不再接收新的任务
        pool.close()

        # 等待所有进程完成
        pool.join()

        # 打印处理数据所花费的时间（以分钟为单位）
        print("Process data took {:.4} minutes.".format((time.time() - start_time) / 60.0))


# 当该脚本作为主程序运行时，执行main函数
if __name__ == '__main__':
    main()
