import os  # 导入os模块，用于文件和目录操作
import argparse  # 导入argparse模块，用于解析命令行参数
import numpy as np  # 导入numpy模块，用于数值计算
import json  # 导入json模块，用于读写JSON文件
from tqdm import tqdm  # 导入tqdm模块，用于在循环中显示进度条
import torch  # 导入torch模块，用于加载点云数据

def main():
    parser = argparse.ArgumentParser()  # 创建一个命令行参数解析器
    parser.add_argument('pcd_data_dir')  # 添加点云数据目录参数
    parser.add_argument('out_file')  # 添加输出文件参数
    args = parser.parse_args()  # 解析命令行参数

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)  # 创建输出文件的目录

    scan_ids = [x.split('.')[0] for x in os.listdir(args.pcd_data_dir)]  # 获取所有扫描ID
    scan_ids.sort()  # 对扫描ID进行排序

    scene_locs = {}  # 初始化一个字典，用于存储场景的位置信息

    # 遍历所有扫描ID，并使用tqdm显示进度
    for scan_id in tqdm(scan_ids):
        # 加载点云数据（包括点、颜色、语义标签和实例标签）
        points, colors, _, inst_labels = torch.load(
            os.path.join(args.pcd_data_dir, '%s.pth'%scan_id)
        )
        # 计算场景中心点（所有点的平均值）
        scene_center = np.mean(points, 0)
        # 计算场景大小（最大值和最小值之间的差）
        scene_size = points.max(0) - points.min(0)
        # 将中心点和大小连接起来，并转换为列表格式
        scene_locs[scan_id] = np.concatenate([scene_center, scene_size], 0).tolist()

    # 将场景位置信息保存到JSON文件
    json.dump(scene_locs, open(os.path.join(args.out_file), 'w'))

# 如果此脚本是直接运行的，则调用main函数
if __name__ == '__main__':
    main()
