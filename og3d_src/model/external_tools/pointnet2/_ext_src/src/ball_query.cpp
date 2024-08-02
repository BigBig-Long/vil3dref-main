// 版权声明，表明这段代码的版权归属和许可协议
// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// 包含自定义的头文件，这些头文件可能定义了一些辅助函数和内核声明
#include "ball_query.h"
#include "utils.h"

// 声明一个C++函数，该函数将由CUDA内核调用，用于点球查询操作
// 参数包括：批量大小b，输入点的数量n，输出点的数量m，搜索半径radius，邻居点的数量nsample，以及输入和输出的指针
void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
int nsample, const float *new_xyz,
const float *xyz, int *idx);

// 定义一个名为ball_query的函数，它是一个Tensor操作，用于在给定的搜索半径内查询点的邻居索引
at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
const int nsample) {
    // 检查输入张量是否连续（即内存是否连续），这对于CUDA操作是必要的
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);
    // 检查输入张量是否为浮点类型
    CHECK_IS_FLOAT(new_xyz);
    CHECK_IS_FLOAT(xyz);
    // 检查new_xyz是否在CUDA上，即是否为GPU上的张量
    if (new_xyz.type().is_cuda()) {
        // 如果是，检查xyz张量是否也在CUDA上
        CHECK_CUDA(xyz);
    }

    // 创建一个输出张量，用于存储查询到的索引，其大小为批大小*点数*邻居点数，数据类型为整数
    at::Tensor idx =
        torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                     at::device(new_xyz.device()).dtype(at::ScalarType::Int));

    // 如果输入张量在CUDA上，调用CUDA内核执行实际的点球查询操作
    if (new_xyz.type().is_cuda()) {
        // 调用之前声明的C++函数，该函数会进一步调用CUDA内核
        // 这里传递了必要的参数，包括点的数量、搜索半径、采样点数以及数据指针
        query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                        radius, nsample, new_xyz.data<float>(),
                                        xyz.data<float>(), idx.data<int>());
    } else {
        // 如果输入不在CUDA上，抛出一个错误，因为当前实现不支持CPU
        TORCH_CHECK(false, "CPU not supported");
    }

    // 返回查询结果张量
    return idx;
}
