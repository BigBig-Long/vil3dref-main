// 版权和许可协议声明，与之前相同

#include "group_points.h" // 包含分组点云操作的头文件
#include "utils.h"        // 包含辅助函数和宏定义的头文件

// 声明CUDA内核函数的包装器，这些函数的具体实现在CUDA内核文件中
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
const float *points, const int *idx,
float *out);
void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
int nsample, const float *grad_out,
const int *idx, float *grad_points);

// 实现group_points函数，用于在GPU上对点云进行分组
at::Tensor group_points(at::Tensor points, at::Tensor idx) {
    // 检查points和idx张量是否连续（在内存中连续存储）
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    // 检查points是否为浮点类型，idx是否为整型
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);
    // 检查idx是否也在CUDA设备上
    if (points.type().is_cuda()) {
        CHECK_CUDA(idx);
    }
    // 初始化输出张量，所有元素设为0
    at::Tensor output =
        torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                     at::device(points.device()).dtype(at::ScalarType::Float));
    // 如果在CUDA设备上，调用CUDA内核执行操作
    if (points.type().is_cuda()) {
        group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                   idx.size(1), idx.size(2), points.data<float>(),
                                   idx.data<int>(), output.data<float>());
    } else {
        // 如果不是CUDA设备，抛出错误，因为CPU不支持
        TORCH_CHECK(false, "CPU not supported");
    }
    // 返回输出张量
    return output;
}

// 实现group_points_grad函数，用于在GPU上计算group_points操作的梯度
at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
    // 类似地，检查grad_out和idx张量是否连续，类型是否正确
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);
    // 检查grad_out是否在CUDA设备上
    if (grad_out.type().is_cuda()) {
        CHECK_CUDA(idx);
    }
    // 初始化梯度输出张量
    at::Tensor output =
        torch::zeros({grad_out.size(0), grad_out.size(1), n},
                     at::device(grad_out.device()).dtype(at::ScalarType::Float));
    // 如果在CUDA设备上，调用CUDA内核执行反向传播操作
    if (grad_out.type().is_cuda()) {
        group_points_grad_kernel_wrapper(
            grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
            grad_out.data<float>(), idx.data<int>(), output.data<float>());
    } else {
        // 如果不是CUDA设备，抛出错误，因为CPU不支持
        TORCH_CHECK(false, "CPU not supported");
    }
    // 返回梯度输出张量
    return output;
}
