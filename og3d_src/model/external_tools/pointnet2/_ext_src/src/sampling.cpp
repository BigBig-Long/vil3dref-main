// 版权声明，表明这段代码的归属和许可协议
// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "sampling.h" // 包含自定义的采样函数头文件
#include "utils.h"     // 包含一些辅助工具函数的头文件

// CUDA内核函数声明，这些函数的具体实现在其他地方定义
void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
    const float *points, const int *idx, float *out); // 聚集点的内核函数
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
    const float *grad_out, const int *idx, float *grad_points); // 聚集点梯度的内核函数
void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
    const float *dataset, float *temp, int *idxs); // 最远点采样的内核函数

// 定义一个PyTorch操作，用于根据索引聚集点
at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
    CHECK_CONTIGUOUS(points); // 检查points是否是连续的内存
    CHECK_CONTIGUOUS(idx);    // 检查idx是否是连续的内存
    CHECK_IS_FLOAT(points);   // 检查points是否是浮点类型
    CHECK_IS_INT(idx);        // 检查idx是否是整型
    if (points.type().is_cuda()) { // 如果points是CUDA类型
        CHECK_CUDA(idx);       // 检查idx是否也是CUDA类型
    }
    // 创建输出张量，大小为[points.size(0), points.size(1), idx.size(1)]
    at::Tensor output = torch::zeros({points.size(0), points.size(1), idx.size(1)},
                                     at::device(points.device()).dtype(at::ScalarType::Float));
    if (points.type().is_cuda()) { // 如果points是CUDA类型
        // 调用CUDA内核函数，执行实际的聚集操作
        gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                    idx.size(1), points.data<float>(), idx.data<int>(), output.data<float>());
    } else {
        TORCH_CHECK(false, "CPU not supported"); // 如果不是CUDA，抛出错误，因为CPU不支持
    }
    return output; // 返回结果张量
}

// 定义一个PyTorch操作，用于计算聚集点的梯度
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
    CHECK_CONTIGUOUS(grad_out); // 检查grad_out是否是连续的内存
    CHECK_CONTIGUOUS(idx);      // 检查idx是否是连续的内存
    CHECK_IS_FLOAT(grad_out);   // 检查grad_out是否是浮点类型
    CHECK_IS_INT(idx);          // 检查idx是否是整型
    if (grad_out.type().is_cuda()) { // 如果grad_out是CUDA类型
        CHECK_CUDA(idx);         // 检查idx是否也是CUDA类型
    }
    // 创建输出张量，大小为[grad_out.size(0), grad_out.size(1), n]
    at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), n},
                                     at::device(grad_out.device()).dtype(at::ScalarType::Float));
    if (grad_out.type().is_cuda()) { // 如果grad_out是CUDA类型
        // 调用CUDA内核函数，计算聚集点的梯度
        gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                         idx.size(1), grad_out.data<float>(), idx.data<int>(), output.data<float>());
    } else {
        TORCH_CHECK(false, "CPU not supported"); // 如果不是CUDA，抛出错误，因为CPU不支持
    }
    return output; // 返回梯度张量
}

// 定义一个PyTorch操作，用于最远点采样
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
    CHECK_CONTIGUOUS(points); // 检查points是否是连续的内存
    CHECK_IS_FLOAT(points);   // 检查points是否是浮点类型
    // 创建输出张量，大小为[points.size(0), nsamples]
    at::Tensor output = torch::zeros({points.size(0), nsamples},
                                                                          at::device(points.device()).dtype(at::ScalarType::Int));
    // 创建一个临时张量，用于最远点采样算法中的距离计算
    at::Tensor tmp = torch::full({points.size(0), points.size(1)}, 1e10,
                                 at::device(points.device()).dtype(at::ScalarType::Float));
    if (points.type().is_cuda()) { // 如果points是CUDA类型
        // 调用CUDA内核函数，执行最远点采样操作
        furthest_point_sampling_kernel_wrapper(
            points.size(0), points.size(1), nsamples, points.data<float>(),
            tmp.data<float>(), output.data<int>());
    } else {
        TORCH_CHECK(false, "CPU not supported"); // 如果不是CUDA，抛出错误，因为CPU不支持
    }
    return output; // 返回采样后的索引张量
}

// 以下是对CUDA编程相关语法的解释：

- `void gather_points_kernel_wrapper(...)`: 这是一个CUDA内核函数的声明，`void`表示它没有返回值。这些内核函数通常在GPU上并行执行，处理大量的数据。

- `at::Tensor`: 这是PyTorch中张量的类，用于表示多维数组。

- `points.data<float>()`: 这是PyTorch中获取张量数据的指针的方法，`<float>`指定了指针指向的数据类型。

- `CHECK_CONTIGUOUS`, `CHECK_IS_FLOAT`, `CHECK_IS_INT`, `CHECK_CUDA`: 这些是宏定义，用于检查输入张量的属性，如是否是连续的内存、数据类型是否正确、是否是CUDA张量。

- `torch::zeros`, `torch::full`: 这些是PyTorch C++ API中的函数，用于创建新的张量，分别创建一个填充了0和张量全为指定值的张量。

- `at::device(points.device())`: 这是从输入张量获取设备信息，确保输出张量与输入张量在同一个设备上。

- ` TORCH_CHECK(false, "CPU not supported")`: 这是一个宏，用于抛出错误，如果条件为假（在这种情况下，如果代码运行在CPU上），则会抛出带有给定消息的错误。

- `furthest_point_sampling_kernel_wrapper`: 这是执行最远点采样算法的CUDA内核函数。

在CUDA编程中，内核函数通常在GPU上并行执行，处理大量的数据。这些函数通过特定的CUDA API调用，并且通常需要指定执行配置（如线程块大小和网格大小）。在上述代码中，这些配置是通过内核包装器函数（如`gather_points_kernel_wrapper`）来隐式处理的，这些包装器函数的具体实现在其他文件中定义。

