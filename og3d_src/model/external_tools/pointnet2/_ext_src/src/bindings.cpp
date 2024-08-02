// 版权声明，表示这段代码的版权归属Facebook公司及其关联公司。

// 许可协议声明，说明这段代码遵循的是MIT许可协议，协议的具体内容在源码根目录的LICENSE文件中。

#include "ball_query.h"  // 包含ball_query相关的函数和类声明，这是一个自定义的头文件
#include "group_points.h" // 包含group_points相关的函数和类声明，这也是一个自定义的头文件
#include "interpolate.h"  // 包含插值相关的函数和类声明，同样是一个自定义的头文件
#include "sampling.h"     // 包含采样相关的函数和类声明，也是一个自定义的头文件

// PYBIND11_MODULE宏用于创建一个Pybind11模块，这是将C++代码绑定到Python的库
// TORCH_EXTENSION_NAME是一个宏，通常用于指定扩展模块的名称
// m是一个模块对象，用于向Python绑定C++函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def()用于将C++函数绑定到Python模块中的指定名称
    // "gather_points"是Python中使用的函数名
    // &gather_points是C++中函数的地址，这里假设gather_points是一个已经定义好的C++函数
    m.def("gather_points", &gather_points); // 绑定一个函数，用于收集点云数据

    // 以下行与上面的类似，都是将C++函数绑定到Python模块
    m.def("gather_points_grad", &gather_points_grad); // 绑定gather_points的梯度计算函数，用于反向传播
    m.def("furthest_point_sampling", &furthest_point_sampling); // 绑定一个函数，用于最远点采样
    m.def("three_nn", &three_nn); // 绑定一个函数，用于寻找每个点的三个最近邻
    m.def("three_interpolate", &three_interpolate); // 绑定一个函数，用于三线性插值
    m.def("three_interpolate_grad", &three_interpolate_grad); // 绑定three_interpolate的梯度计算函数，用于反向传播
    m.def("ball_query", &ball_query); // 绑定一个函数，用于执行球查询操作，查找点云中特定半径内的点
    m.def("group_points", &group_points); // 绑定一个函数，用于分组点云中的点
    m.def("group_points_grad", &group_points_grad); // 绑定group_points的梯度计算函数，用于反向传播
}
