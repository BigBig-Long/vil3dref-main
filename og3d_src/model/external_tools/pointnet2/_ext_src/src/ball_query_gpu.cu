// 版权声明和许可协议，与之前相同

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"
// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
// CUDA内核声明，它将在GPU上并行运行
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
int nsample,
const float *__restrict__ new_xyz,
const float *__restrict__ xyz,
int *__restrict__ idx) {
    // 获取当前块的索引，对应于批处理中的哪一个批次
    int batch_index = blockIdx.x;
    // 根据批次索引调整xyz和new_xyz的指针，以便指向当前批次的起始位置
    xyz += batch_index * n * 3;
    new_xyz += batch_index * m * 3;
    // 根据批次索引调整idx的指针，以便写入当前批次的结果
    idx += m * nsample * batch_index;
    // 获取线程索引和步长（每个线程块中的线程数）
    int index = threadIdx.x;
    int stride = blockDim.x;
    // 计算半径的平方，用于距离比较
    float radius2 = radius * radius;
    // 并行处理每个new_xyz点
    for (int j = index; j < m; j += stride) {
        // 获取new_xyz点的坐标
        float new_x = new_xyz[j * 3 + 0];
        float new_y = new_xyz[j * 3 + 1];
        float new_z = new_xyz[j * 3 + 2];
        // 用于存储找到的邻近点的计数器
        int cnt = 0;
        // 遍历所有xyz点，寻找邻近点
        for (int k = 0; k < n && cnt < nsample; ++k) {
            // 获取xyz点的坐标
            float x = xyz[k * 3 + 0];
            float y = xyz[k * 3 + 1];
            float z = xyz[k * 3 + 2];
            // 计算两点之间的距离平方
            float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                       (new_z - z) * (new_z - z);
            // 如果距离小于半径，则认为这是一个邻近点
            if (d2 < radius2) {
                // 如果是第一个找到的邻近点，初始化idx数组
                if (cnt == 0) {
                    for (int l = 0; l < nsample; ++l) {
                        idx[j * nsample + l] = k;
                    }
                }
                // 存储邻近点的索引
                idx[j * nsample + cnt] = k;
                // 增加计数器
                ++cnt;
            }
        }
    }
}

// CUDA内核的包装函数，用于在CPU端调用CUDA内核
void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
int nsample, const float *new_xyz,
const float *xyz, int *idx) {
    // 获取当前CUDA流的句柄
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 启动CUDA内核，指定了块的数量（等于批处理大小b），线程块的大小（由opt_n_threads(m)决定），以及使用的流
    query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
        b, n, m, radius, nsample, new_xyz, xyz, idx);
    // 检查CUDA调用是否有错误发生
    CUDA_CHECK_ERRORS();
}
