# Copyright (c) Facebook, Inc. and its affiliates.
# 版权声明，表明这段代码是由Facebook公司及其关联公司拥有。

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 这段源代码在MIT许可证下授权，该许可证可以在源代码树的根目录下的LICENSE文件中找到。

''' Testing customized ops. '''
# 这是一个文档字符串，说明这个脚本的目的是测试自定义操作（customized ops）。

import torch
# 导入PyTorch库，一个流行的开源机器学习库。

from torch.autograd import gradcheck
# 从torch.autograd模块导入gradcheck函数，该函数用于检查自动微分中的梯度计算是否正确。

import numpy as np
# 导入NumPy库，用于进行科学计算。

import os
# 导入os模块，它提供了与操作系统交互的功能。

import sys
# 导入sys模块，它提供了一些变量和函数，用来操纵Python运行时环境。

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取当前执行文件的绝对路径的目录部分，并赋值给BASE_DIR变量。

sys.path.append(BASE_DIR)
# 将BASE_DIR添加到Python的模块搜索路径中，这样就可以导入该目录下的模块了。

import pointnet2_utils
# 导入pointnet2_utils模块，这可能是包含了点云处理相关功能的自定义模块。

def test_interpolation_grad():
# 定义一个名为test_interpolation_grad的函数，用于测试插值函数的梯度。

    batch_size = 1
    # 定义一个变量，表示批处理大小，这里设为1。

    feat_dim = 2
    # 定义一个变量，表示特征维度，这里设为2。

    m = 4
    # 定义一个变量m，这里设为4，但未说明具体含义，可能是指某种维度或点数。

    feats = torch.randn(batch_size, feat_dim, m, requires_grad=True).float().cuda()
    # 创建一个形状为[batch_size, feat_dim, m]的随机张量，并设置requires_grad=True，以便计算梯度。
    # 然后，将其转换为float类型并移动到GPU上。

    def interpolate_func(inputs):
    # 定义一个名为interpolate_func的内嵌函数，用于进行插值操作。

        idx = torch.from_numpy(np.array([[[0,1,2],[1,2,3]]])).int().cuda()
        # 创建一个索引张量，从NumPy数组转换而来，并设置为整型，然后移动到GPU上。

        weight = torch.from_numpy(np.array([[[1,1,1],[2,2,2]]])).float().cuda()
        # 创建一个权重张量，同样从NumPy数组转换而来，并设置为浮点型，然后移动到GPU上。

        interpolated_feats = pointnet2_utils.three_interpolate(inputs, idx, weight)
        # 使用自定义模块pointnet2_utils中的three_interpolate函数进行三线性插值。

        return interpolated_feats
        # 返回插值结果。

    assert (gradcheck(interpolate_func, feats, atol=1e-1, rtol=1e-1))
    # 使用gradcheck函数检查interpolate_func函数在feats输入上的梯度计算是否正确。
    # atol和rtol分别代表绝对和相对容差。

if __name__ == '__main__':
    test_interpolation_grad()
    # 如果这个脚本是直接运行的，而不是被导入到其他脚本中，则运行test_interpolation_grad函数。
