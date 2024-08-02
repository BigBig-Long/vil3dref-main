// 版权和许可协议声明，与之前相同

#include <stdio.h> // 包含标准输入输出头文件
#include <stdlib.h> // 包含C标准库头文件，提供了一些基础的函数

#include "cuda_utils.h" // 包含CUDA相关的辅助函数和宏定义

// 定义group_points_kernel CUDA内核函数
// 输入参数：points(b, c, n) idx(b, npoints, nsample)
// 输出参数：out(b, c, npoints, nsample)
__global__ void group_points_kernel(int b, int c, int n, int npoints, int nsample,
const float *__restrict__ points,
const int *__restrict__ idx,
float *__restrict__ out) {
    // 获取当前线程所在的批次索引
    int batch_index = blockIdx.x;
    // 根据批次索引计算每个批次数据在points和idx中的起始地址
    points += batch_index * n * c;
    idx += batch_index * npoints * nsample;
    out += batch_index * npoints * nsample * c;
    // 计算当前线程的全局索引
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    // 计算每个线程块内所有线程的总数
    const int stride = blockDim.y * blockDim.x;
    // 循环遍历每个点云的所有点
    for (int i = index; i < c * npoints; i += stride) {
        // 计算当前点的通道索引l和点云索引j
        const int l = i / npoints;
        const int j = i % npoints;
        // 对于每个点，遍历其所有的采样点
        for (int k = 0; k < nsample; ++k) {
            // 根据索引idx获取采样点的原始索引ii
            int ii = idx[j * nsample + k];
            // 将采样点的值赋给输出张量
            out[(l * npoints + j) * nsample + k] = points[l * n + ii];
        }
    }
}

// 定义group_points_kernel的C++包装器函数
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
const float *points, const int *idx,
float *out) {
    // 获取当前CUDA流的句柄
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 启动CUDA内核，指定了grid和block的大小，以及使用的流
    group_points_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
        b, c, n, npoints, nsample, points, idx, out);
    // 检查CUDA操作是否成功
    CUDA_CHECK_ERRORS();
}

// 定义group_points_grad_kernel CUDA内核函数
// 输入参数：grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// 输出参数：grad_points(b, c, n)
__global__ void group_points_grad_kernel(int b, int c, int n, int npoints, int nsample,
const float *__restrict__ grad_out,
const int *__restrict__ idx,
float *__restrict__ grad_points) {
    // 获取当前线程所在的批次索引
    int batch_index = blockIdx.x;
    // 根据批次索引计算每个批次数据在grad_out和idx中的起始地址
    grad_out += batch_index * npoints * nsample * c;
    idx += batch_index * npoints * nsample;
    grad_points += batch_index * n * c;
    // 计算当前线程的全局索引
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    // 计算每个线程块内所有线程的总数
    const int stride = blockDim.y * blockDim.x;
    // 循环遍历每个点云的所有点
    for (int i = index; i < c * npoints; i += stride) {
        // 计算当前点的通道索引l和点云索引j
        const int l = i / npoints;
        const int j = i % npoints;
        // 对于每个点，遍历其所有的采样点
        for (int k = 0; k < nsample; ++k) {
            // 根据索引idx获取采样点的原始索引ii
            int ii = idx[j * nsample + k];
            // 使用原子操作累加梯度到grad_points
            // 这确保了在多个线程尝试更新同一个元素时的线程安全
            atomicAdd(grad_points + l * n + ii,
                      grad_out[(l * npoints + j) * nsample + k]);
        }
    }
}

// 定义group_points_grad_kernel的C++包装器函数
void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points) {
    // 获取当前CUDA流的句柄
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 启动CUDA内核，指定了grid和block的大小，以及使用的流
    group_points_grad_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
        b, c, n, npoints, nsample, grad_out, idx, grad_points);
    // 检查CUDA操作是否成功
    CUDA_CHECK_ERRORS();
}
