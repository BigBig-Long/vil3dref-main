// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <stdio.h> // 包含标准输入输出头文件
#include <stdlib.h> // 包含标准库头文件，提供了一些基础的函数
#include "cuda_utils.h" // 包含一些CUDA相关的辅助工具函数

// CUDA内核函数声明，用于根据索引聚集点
// 输入参数：points(b, c, n) 表示一个三维张量，b表示batch size，c表示channel数量，n表示每个channel的点数
// 输入参数：idx(b, m) 表示一个二维张量，存储了每个batch中要聚集的点的索引
// 输出参数：out(b, c, m) 表示输出的三维张量，包含了聚集后的点
__global__ void gather_points_kernel(int b, int c, int n, int m,
    const float *__restrict__ points, // 限制指针，告诉编译器只在这个函数内部使用这个指针
    const int *__restrict__ idx,      // 同上，限制指针，提高内存访问效率
    float *__restrict__ out) {        // 同上，限制指针，用于输出
    // blockIdx.x和blockIdx.y分别表示当前线程块在grid中的x和y坐标
    // gridDim.x和gridDim.y分别表示grid在x和y方向上的块数量
    for (int i = blockIdx.x; i < b; i += gridDim.x) { // 遍历每个batch
        for (int l = blockIdx.y; l < c; l += gridDim.y) { // 遍历每个channel
            for (int j = threadIdx.x; j < m; j += blockDim.x) { // 遍历每个要聚集的点
                // 计算索引，从idx中获取当前点的索引
                int a = idx[i * m + j];
                // 根据索引从points中取出点，并存储到out中
                out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
            }
        }
    }
}

// CUDA内核函数的包装器，用于设置执行配置并调用内核
void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
    const float *points, const int *idx,
    float *out) {
    // 启动CUDA内核
    // dim3(b, c, 1)定义了grid的大小，即在每个维度上分别有多少个block
    // opt_n_threads(npoints)是一个辅助函数，用于计算每个block的最佳线程数
    // at::cuda::getCurrentCUDAStream()获取当前CUDA流的句柄，用于同步内核执行
    gather_points_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                           at::cuda::getCurrentCUDAStream()>>>(b, c, n, npoints,
                                                                points, idx, out);
    // 检查CUDA执行过程中是否有错误发生
    CUDA_CHECK_ERRORS();
}
// 输入参数：grad_out(b, c, m) 表示输出的梯度，b是batch size，c是channel数量，m是采样点的数量
// 输入参数：idx(b, m) 表示采样点的索引
// 输出参数：grad_points(b, c, n) 表示输入点的梯度，n是原始点的数量
__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
    const float *__restrict__ grad_out, // 梯度输出
    const int *__restrict__ idx,        // 采样点索引
    float *__restrict__ grad_points) {  // 输入点的梯度
    // 类似于前面的遍历方式，对每个batch和channel进行遍历
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int l = blockIdx.y; l < c; l += gridDim.y) {
            for (int j = threadIdx.x; j < m; j += blockDim.x) {
                // 获取当前采样点的索引
                int a = idx[i * m + j];
                // 使用原子操作累加梯度到对应的位置
                // atomicAdd是CUDA提供的原子操作函数，确保多个线程更新同一个内存地址时的线程安全
                atomicAdd(grad_points + (i * c + l) * n + a,
                          grad_out[(i * c + l) * m + j]);
            }
        }
    }
}

// 梯度计算内核的包装器函数
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
    const float *grad_out, const int *idx,
    float *grad_points) {
    // 启动梯度计算内核
    gather_points_grad_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                                at::cuda::getCurrentCUDAStream()>>>(
        b, c, n, npoints, grad_out, idx, grad_points);
    // 检查CUDA执行过程中的错误
    CUDA_CHECK_ERRORS();
}

// 定义一个设备函数，用于更新距离和索引
// 这个函数可能是用于其他计算，如计算点之间的距离
__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
    // 获取两个点的距离和索引
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    // 更新距离和索引，保留较大的距离和对应的索引
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1; // 如果v2更大，更新索引为i2，否则保持i1
}
// 使用模板定义一个CUDA内核函数，block_size是线程块中线程的数量
template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, // b: batch size, n: 每个batch中的点数, m: 要采样的点数
    const float *__restrict__ dataset, // 输入数据集，包含点的坐标 (b, n, 3)
    float *__restrict__ temp, // 临时存储空间，用于存储距离 (b, n)
    int *__restrict__ idxs // 输出索引，存储采样点的索引 (b, m)
) {
    // 如果要采样的点数m为0或负数，则直接返回
    if (m <= 0) return;

    // 在共享内存中声明两个数组，用于存储距离和对应的索引
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    // 获取当前线程块处理的batch索引
    int batch_index = blockIdx.x;

    // 根据batch索引调整数据集、临时存储和输出索引的指针
    dataset += batch_index * n * 3; // 跳到当前batch的数据开始位置
    temp += batch_index * n; // 跳到当前batch的临时存储开始位置
    idxs += batch_index * m; // 跳到当前batch的输出索引开始位置

    // 获取当前线程的索引
    int tid = threadIdx.x;

    // 定义线程步长，用于在数据集中进行遍历
    const int stride = block_size;

    // 定义一个变量存储上一个采样点的索引
    int old = 0;

    // 如果当前线程是线程块中的第一个线程，则将第一个采样点的索引设置为0
    if (threadIdx.x == 0) idxs[0] = old;

    // 同步所有线程，确保第一个采样点的索引被正确设置
    __syncthreads();

    // 对剩余的采样点进行循环
    for (int j = 1; j < m; j++) {
        // 初始化当前线程找到的最远点及其距离
        int besti = 0;
        float best = -1;

        // 获取上一个采样点的坐标
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];

        // 遍历所有点，找到距离上一个采样点最远的点
        for (int k = tid; k < n; k += stride) {
            // 获取当前点的坐标
            float x2 = dataset[k * 3 + 0];
            float y2 = dataset[k * 3 + 1];
            float z2 = dataset[k * 3 + 2];

            // 计算当前点的模长，如果太小则跳过
            float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            if (mag <= 1e-3) continue;

            // 计算当前点到上一个采样点的距离
            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

            // 更新临时存储中的距离，取当前距离和之前存储的距离的较小值
            float d2 = min(d, temp[k]);
            temp[k] = d2;

            // 如果当前点距离更远，则更新最远点和距离
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }

        // 将当前线程找到的最远点和距离存储到共享内存中
        dists[tid] = best;
        dists_i[tid] = besti;

        // 同步所有线程
        __syncthreads();

        // 以下代码使用二进制减少技术找到所有线程中的最远点
        // 根据线程块大小进行多次比较和同步，以确定哪个点的距离最远
        // 每次比较后，只有一半的线程会继续参与下一轮比较
        if (block_size >= 512) {
            if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
            }
            if (block_size >= 256) {
            if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
            }
            if (block_size >= 128) {
            if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
            }
            if (block_size >= 64) {
            if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
            }
            if (block_size >= 32) {
            if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
            }
            if (block_size >= 16) {
            if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
            }
            if (block_size >= 8) {
            if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
            }
            if (block_size >= 4) {
            if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
            }
            if (block_size >= 2) {
            if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
            }
    // 更新上一个采样点的索引为找到的最远点的索引
    old = dists_i[0];

    // 如果当前线程是线程块中的第一个线程，则将新的采样点索引存储到输出数组中
    if (tid == 0) idxs[j] = old;
}

// **语法说明：**
// - `__global__`：声明一个可以在设备上执行的内核函数。
// - `template <unsigned int block_size>`：使用模板定义线程块大小。
// - `__restrict__`：告诉编译器，指针是唯一的，没有其他指针指向相同的内存区域。
// - `__shared__`：声明共享内存变量，这些变量在同一个线程块中的所有线程之间共享。
// - `blockIdx.x`：当前线程块的索引。
// - `threadIdx.x`：当前线程在其块内的索引。
// - `__syncthreads()`：同步同一个线程块中的所有线程。
// - `__update`：这应该是一个自定义函数，用于更新共享内存中的距离和索引，但在这个代码片段中没有给出定义。
//
// 请注意，这段代码中的`__update`函数没有定义，它应该是用来比较两个线程的距离，并更新共享内存中的最远点和距离。这个函数是实现二进制减少的关键部分。



// 定义一个包装器函数furthest_point_sampling_kernel_wrapper，接收以下参数：
// b: 批次大小，n: 输入点的数量，m: 要采样的点的数量，
// dataset: 存储点云数据的指针，temp: 临时存储空间，idxs: 存储采样点索引的指针
void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
const float *dataset, float *temp,
int *idxs) {
    // 获取适用于当前输入点数量的最佳线程数
    unsigned int n_threads = opt_n_threads(n);
    // 获取当前CUDA流的引用
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 使用switch语句根据线程数选择合适的核函数版本
    switch (n_threads) {
        // 对于每个case，根据线程块大小调用不同的特殊化版本的核函数
        case 512:
            furthest_point_sampling_kernel<512>
            <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); //<<<Dg, Db, 0, stream>>>
            break; // 其中Dg是grid大小，Db是block大小，0是共享内存大小，stream是CUDA流
        case 256:
            furthest_point_sampling_kernel<256>
            <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 128:
          furthest_point_sampling_kernel<128>
              <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
          break;
        case 64:
          furthest_point_sampling_kernel<64>
              <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
          break;
        case 32:
          furthest_point_sampling_kernel<32>
              <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
          break;
        case 16:
          furthest_point_sampling_kernel<16>
              <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
          break;
        case 8:
          furthest_point_sampling_kernel<8>
              <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
          break;
        case 4:
          furthest_point_sampling_kernel<4>
              <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
          break;
        case 2:
          furthest_point_sampling_kernel<2>
              <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
          break;
        case 1:
          furthest_point_sampling_kernel<1>
              <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
          break;
        // 默认情况下使用512线程块的版本
        default:
            furthest_point_sampling_kernel<512>
            <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    }
    // 检查CUDA操作是否发生错误
    CUDA_CHECK_ERRORS();
}
