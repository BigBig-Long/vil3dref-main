import os  # 导入os模块，用于操作系统相关的操作
import json  # 导入json模块，用于处理JSON数据
import glob  # 导入glob模块，用于从目录中获取文件列表
import torch  # 导入torch模块，用于处理PyTorch张量
import numpy as np  # 导入numpy模块，并重命名为np，用于数学和科学计算
from sklearn.mixture import GaussianMixture  # 从sklearn库中导入GaussianMixture类，用于高斯混合模型

# 设置扫描数据所在的目录
scan_dir = 'datasets/referit3d/exprs_neurips22/instance_segm/pointgroup/scan_data'
# 设置输出目录，该目录将存储每个实例ID对应的GMM颜色聚类结果
output_dir = os.path.join(scan_dir, 'instance_id_to_gmm_color')
# 创建输出目录，如果目录已存在则忽略错误
os.makedirs(output_dir, exist_ok=True)

# 遍历指定目录下的每个扫描文件
for scan_file in glob.glob(os.path.join(scan_dir, 'pcd_with_global_alignment', '*')):
    # 从文件名中提取扫描ID（去除文件扩展名）
    scan_id = os.path.basename(scan_file).split('.')[0]
    print(scan_file)  # 打印文件名，用于调试目的

    # 加载点云数据（xyz坐标、rgb颜色、语义标签、实例标签）
    data = torch.load(scan_file)
    colors = data[1]  # 提取颜色数据
    instance_labels = data[3]  # 提取实例标签数据

    # 如果没有实例标签，则跳过当前扫描文件
    if instance_labels is None:
        continue

    # 归一化颜色数据（将RGB值范围从[0, 255]转换为[-1, 1]）
    colors = colors / 127.5 - 1

    # 初始化一个列表，用于存储每个实例的聚类颜色
    clustered_colors = []

    # 遍历每个实例标签
    for i in range(instance_labels.max() + 1):
        # 创建一个掩码，用于选择当前实例的颜色数据
        mask = instance_labels == i  # 这一步可能对于大型数据集来说比较耗时

        # 提取当前实例的颜色数据
        obj_colors = colors[mask]

        # 使用高斯混合模型对颜色数据进行聚类，假设有3个组件（对应RGB三个通道）
        gm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(obj_colors)

        # 将GMM的权重和均值添加到clustered_colors列表中
        clustered_colors.append({
            'weights': gm.weights_.tolist(),  # 混合权重
            'means': gm.means_.tolist(),  # 混合均值（即聚类中心）
        })

    # 将聚类颜色结果以JSON格式保存，文件名为对应的扫描ID
    json.dump(
        clustered_colors,
        open(os.path.join(output_dir, '%s.json' % scan_id), 'w')
    )

