import os
import json
import glob
import torch
import numpy as np
from sklearn.mixture import GaussianMixture

# 指定扫描数据的目录
scan_dir = 'datasets/referit3d/scan_data'
# 指定输出目录，该目录将存储每个实例ID对应的GMM颜色聚类结果
output_dir = os.path.join(scan_dir, 'instance_id_to_gmm_color')
# 创建输出目录，如果目录已存在则忽略错误
os.makedirs(output_dir, exist_ok=True)

# 遍历每个扫描文件
for scan_file in glob.glob(os.path.join(scan_dir, 'pcd_with_global_alignment', '*')):
    # 获取扫描文件的ID（去除文件扩展名）
    scan_id = os.path.basename(scan_file).split('.')[0]
    print(scan_file)  # 打印当前处理的文件名

    # 加载点云数据，包括xyz坐标、rgb颜色、语义标签和实例标签
    data = torch.load(scan_file)
    colors = data[1]  # 获取颜色数据
    instance_labels = data[3]  # 获取实例标签数据

    # 如果没有实例标签，则跳过当前扫描文件
    if instance_labels is None:
        continue

    # 归一化颜色数据（将RGB值范围从[0, 255]转换为[-1, 1]）
    colors = colors / 127.5 - 1

    # 初始化存储聚类颜色的列表
    clustered_colors = []

    # 遍历每个实例
    for i in range(instance_labels.max() + 1):
        # 获取当前实例的颜色数据
        mask = instance_labels == i  # 这一步可能比较耗时
        obj_colors = colors[mask]

        # 使用高斯混合模型对颜色数据进行聚类
        gm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(obj_colors)

        # 将聚类结果添加到列表中
        clustered_colors.append({
            'weights': gm.weights_.tolist(),  # 混合权重
            'means': gm.means_.tolist(),  # 混合均值（即聚类中心）
        })

    # 将聚类结果保存为JSON文件
    json.dump(
        clustered_colors,
        open(os.path.join(output_dir, '%s.json' % scan_id), 'w')
    )

