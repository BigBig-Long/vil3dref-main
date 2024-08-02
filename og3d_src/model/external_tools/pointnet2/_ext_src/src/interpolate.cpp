// 版权声明，表明这段代码的版权归属和许可协议

// 引入自定义的头文件，可能包含了一些辅助函数和宏定义
#include "interpolate.h"
#include "utils.h"

// 声明CUDA内核函数的C++包装器，具体的实现可能在其他文件中
void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
const float *known, float *dist2, int *idx);
void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
const float *points, const int *idx,
const float *weight, float *out);
void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
const float *grad_out,
const int *idx, const float *weight,
float *grad_points);

// three_nn函数实现了查找每个未知点的三个最近邻点的功能
std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows) {
    // 检查输入张量是否是连续的，即它们的内存是连续分配的
    CHECK_CONTIGUOUS(unknowns);
    CHECK_CONTIGUOUS(knows);
    // 检查输入张量是否是浮点类型
    CHECK_IS_FLOAT(unknowns);
    CHECK_IS_FLOAT(knows);
    // 如果未知点的类型是CUDA，则检查已知点也是在CUDA上
    if (unknowns.type().is_cuda()) {
        CHECK_CUDA(knows);
    }
    // 初始化输出索引张量和距离平方张量
    at::Tensor idx = torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                                  at::device(unknowns.device()).dtype(at::ScalarType::Int));
    at::Tensor dist2 = torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                                    at::device(unknowns.device()).dtype(at::ScalarType::Float));
    // 如果在CUDA上，调用CUDA内核函数
    if (unknowns.type().is_cuda()) {
        three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                                unknowns.data<float>(), knows.data<float>(),
                                dist2.data<float>(), idx.data<int>());
    } else {
        // 如果不在CUDA上，抛出错误，因为CPU不支持
        TORCH_CHECK(false, "CPU not supported");
    }
    // 返回距离平方和索引张量
    return {dist2, idx};
}

// three_interpolate函数用于根据索引和权重对点进行三线性插值
at::Tensor three_interpolate(at::Tensor points, at::Tensor idx, at::Tensor weight) {
    // 类似地，检查输入张量是否连续和类型是否正确
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);
    // 检查是否在CUDA上
    if (points.type().is_cuda()) {
        CHECK_CUDA(idx);
        CHECK_CUDA(weight);
    }
    // 初始化输出张量
    at::Tensor output = torch::zeros({points.size(0), points.size(1), idx.size(1)},
                                     at::device(points.device()).dtype(at::ScalarType::Float));
    // 如果在CUDA上，调用CUDA内核函数
    if (points.type().is_cuda()) {
        three_interpolate_kernel_wrapper(
            points.size(0), points.size(1), points.size(2), idx.size(1),
            points.data<float>(), idx.data<int>(), weight.data<float>(),
            output.data<float>());
    } else {
        // 如果不在CUDA上，抛出错误
        TORCH_CHECK(false, "CPU not supported");
    }
    // 返回插值后的输出张量
    return output;
}

// three_interpolate_grad函数用于计算三线性插值的梯度
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m) {
    // 同样地，检查输入张量是否连续和类型是否正确
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);
    // 检查是否在CUDA上
    if (grad_out.type().is_cuda()) {
        CHECK_CUDA(idx);
        CHECK_CUDA(weight);
    }
    // 初始化梯度输出张量
    at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), m},
                                     at::device(grad_out.device()).dtype(at::ScalarType::Float));
    // 如果在CUDA上，调用CUDA内核函数计算梯度
    if (grad_out.type().is_cuda()) {
        three_interpolate_grad_kernel_wrapper(
            grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
            grad_out.data<float>(), idx.data<int>(), weight.data<float>(),
            output.data<float>());
    } else {
        // 如果不在CUDA上，抛出错误
        TORCH_CHECK(false, "CPU not supported");
    }
    // 返回计算得到的梯度张量
    return output;
}

