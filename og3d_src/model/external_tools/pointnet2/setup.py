# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

# 指定扩展源代码的根目录
_ext_src_root = "_ext_src"
# 获取扩展的源文件，包括cpp和cu文件
# cu文件是CUDA代码文件，cpp文件是C++代码文件
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
# 获取扩展所需的头文件
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

# 设置setup函数
setup(
    name='pointnet2',  # 包名称
    ext_modules=[
        # 定义CUDA扩展模块
        CUDAExtension(
            name='pointnet2._ext',  # 扩展名称
            sources=_ext_sources,    # 源文件列表
            # 额外的编译参数
            extra_compile_args={
                "cxx": [  # C++编译器参数
                    "-O2",  # 优化级别
                    "-I{}".format("{}/include".format(_ext_src_root))  # 头文件目录
                ],
                "nvcc": [  # NVIDIA CUDA编译器参数
                    "-O2",  # 优化级别
                    "-I{}".format("{}/include".format(_ext_src_root))  # 头文件目录
                ],
            },
        )
    ],
    # 指定构建扩展的命令类
    cmdclass={
        'build_ext': BuildExtension
    }
)

