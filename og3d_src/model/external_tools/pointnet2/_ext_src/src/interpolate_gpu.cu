// Copyright声明，说明这段代码的版权归属和许可协议。

// 引入C语言的标准库，包括数学函数、标准输入输出和动态内存分配。
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// 引入自定义的CUDA工具函数头文件。
#include "cuda_utils.h"

// CUDA内核函数声明，它接受以下参数：
// b: 批量大小，n: 未知点的数量，m: 已知点的数量
// unknown: 未知点的张量，known: 已知点的张量
// dist2: 存储距离平方的结果张量，idx: 存储最近邻索引的结果张量
__global__ void three_nn_kernel(int b, int n, int m,
const float *__restrict__ unknown,
const float *__restrict__ known,
float *__restrict__ dist2,
int *__restrict__ idx) {
    // 获取当前块的索引，对应于当前处理的批次。
    int batch_index = blockIdx.x;
    // 根据批次索引调整指针到当前批次的数据开始位置。
    unknown += batch_index * n * 3; // 每个点有3个坐标。
    known += batch_index * m * 3; // 每个点有3个坐标。
    dist2 += batch_index * n * 3; // 每个未知点要找到3个最近邻。
    idx += batch_index * n * 3; // 存储每个未知点的3个最近邻的索引。

    // 获取当前线程的索引。
    int index = threadIdx.x;
    // 获取线程块的大小，即每个块中的线程数。
    int stride = blockDim.x;
    // 使用一个循环，使每个线程能够处理多个点。
    for (int j = index; j < n; j += stride) {
        // 获取当前未知点的坐标。
        float ux = unknown[j * 3 + 0];
        float uy = unknown[j * 3 + 1];
        float uz = unknown[j * 3 + 2];
        // 初始化距离平方和索引的最大值，用于寻找最近的三个点。
        double best1 = 1e40, best2 = 1e40, best3 = 1e40;
        int besti1 = 0, besti2 = 0, besti3 = 0;
        // 遍历所有已知点。
        for (int k = 0; k < m; ++k) {
            // 获取当前已知点的坐标。
            float x = known[k * 3 + 0];
            float y = known[k * 3 + 1];
            float z = known[k * 3 + 2];
            // 计算当前未知点和已知点之间的欧氏距离平方。
            float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
            // 更新最近三个点的距离平方和索引。
            if (d < best1) {
                best3 = best2;
                besti3 = besti2;
                best2 = best1;
                besti2 = besti1;
                best1 = d;
                besti1 = k;
            } else if (d < best2) {
                best3 = best2;
                besti3 = besti2;
                best2 = d;
                besti2 = k;
            } else if (d < best3) {
                best3 = d;
                besti3 = k;
            }
        }
        // 将计算出的距离平方和索引存储到输出张量中。
        dist2[j * 3 + 0] = best1;
        dist2[j * 3 + 1] = best2;
        dist2[j * 3 + 2] = best3;
        idx[j * 3 + 0] = besti1;
        idx[j * 3 + 1] = besti2;
        idx[j * 3 + 2] = besti3;
    }
}
// three_nn_kernel_wrapper函数用于包装three_nn_kernel CUDA内核，处理CUDA错误检查。
void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
const float *known, float *dist2, int *idx) {
    // 获取当前CUDA流的引用，用于后续的内核执行。
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 调用three_nn_kernel内核，指定了执行的网格大小（b个块，每个块有opt_n_threads(n)个线程），
    // 没有共享内存，使用之前获取的流。
    // opt_n_threads(n)是一个未定义的函数，应该是用于根据n的值优化线程数的函数。
    three_nn_kernel<<<b, opt_n_threads(n), 0, stream>>>(b, n, m, unknown, known, dist2, idx);
    // 检查CUDA执行过程中是否有错误发生。
    CUDA_CHECK_ERRORS();
}

// three_interpolate_kernel是一个CUDA内核，用于执行三线性插值。
// 输入包括点云数据、索引和权重，输出是插值后的结果。
__global__ void three_interpolate_kernel(int b, int c, int m, int n,
const float *__restrict__ points,
const int *__restrict__ idx,
const float *__restrict__ weight,
float *__restrict__ out) {
    // 获取当前块的索引，对应于当前处理的批次。
    int batch_index = blockIdx.x;
    // 根据批次索引调整指针到当前批次的数据开始位置。
    points += batch_index * m * c; // c是点的维度，m是已知点的数量。
    idx += batch_index * n * 3; // n是未知点的数量，每个未知点有3个最近邻索引。
    weight += batch_index * n * 3; // 每个未知点有3个权重。
    out += batch_index * n * c; // 输出结果，每个批次有n个点，每个点有c个维度。

    // 获取当前线程的索引，这里使用二维线程块结构。
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    // 获取线程块的总大小，即每个块中的线程数。
    const int stride = blockDim.y * blockDim.x;
    // 使用循环，使每个线程能够处理多个插值计算。
    for (int i = index; i < c * n; i += stride) {
        // 计算当前处理的维度索引l和点索引j。
        const int l = i / n; // 维度索引。
        const int j = i % n; // 点索引。

        // 获取当前点的三个权重。
        float w1 = weight[j * 3 + 0];
        float w2 = weight[j * 3 + 1];
        float w3 = weight[j * 3 + 2];
        // 获取当前点的三个最近邻索引。
        int i1 = idx[j * 3 + 0];
        int i2 = idx[j * 3 + 1];
        int i3 = idx[j * 3 + 2];

        // 执行三线性插值计算，并将结果存储到输出张量中。
        out[i] = points[l * m + i1] * w1 + points[l * m + i2] * w2 + points[l * m + i3] * w3;
    }
}
// three_interpolate_kernel_wrapper是three_interpolate_grad_kernel的包装器函数。
void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out) {
    // 获取当前CUDA流。
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 调用three_interpolate_kernel内核，使用opt_block_config来配置每个块的线程数。
    // opt_block_config是一个未定义的函数，可能是用于根据输入大小优化线程块配置的函数。
    three_interpolate_kernel<<<b, opt_block_config(n, c), 0, stream>>>(
        b, c, m, n, points, idx, weight, out);
    // 检查CUDA执行过程中是否有错误发生。
    CUDA_CHECK_ERRORS();
}

// three_interpolate_grad_kernel是一个CUDA内核，用于计算三线性插值的梯度。
// 输入是梯度输出、索引和权重，输出是点的梯度。
__global__ void three_interpolate_grad_kernel(
    int b, int c, int n, int m, const float *__restrict__ grad_out,
    const int *__restrict__ idx, const float *__restrict__ weight,
    float *__restrict__ grad_points) {
    // 获取当前块的批次索引。
    int batch_index = blockIdx.x;
    // 根据批次索引调整指针到当前批次的数据开始位置。
    grad_out += batch_index * n * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    grad_points += batch_index * m * c;
    // 获取当前线程的索引。
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    // 获取线程块的总大小。
    const int stride = blockDim.y * blockDim.x;
    // 循环处理每个输出点的每个维度。
    for (int i = index; i < c * n; i += stride) {
        // 计算当前处理的维度索引和点索引。
        const int l = i / n; // 维度索引。
        const int j = i % n; // 点索引。
        // 获取权重和索引。
        float w1 = weight[j * 3 + 0];
        float w2 = weight[j * 3 + 1];
        float w3 = weight[j * 3 + 2];
        int i1 = idx[j * 3 + 0];
        int i2 = idx[j * 3 + 1];
        int i3 = idx[j * 3 + 2];
        // 使用原子操作累加梯度到对应的三个最近邻点。
        atomicAdd(grad_points + l * m + i1, grad_out[i] * w1);
        atomicAdd(grad_points + l * m + i2, grad_out[i] * w2);
        atomicAdd(grad_points + l * m + i3, grad_out[i] * w3);
    }
}

// three_interpolate_grad_kernel_wrapper是three_interpolate_grad_kernel的包装器函数。
void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points) {
    // 获取当前CUDA流。
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 调用three_interpolate_grad_kernel内核，使用opt_block_config来配置每个块的线程数。
    three_interpolate_grad_kernel<<<b, opt_block_config(n, c), 0, stream>>>(
        b, c, n, m, grad_out, idx, weight, grad_points);
    // 检查CUDA执行过程中是否有错误发生。
    CUDA_CHECK_ERRORS();
}

