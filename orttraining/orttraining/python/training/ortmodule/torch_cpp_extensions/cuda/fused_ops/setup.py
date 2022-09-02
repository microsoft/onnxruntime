# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import fileinput
import os
import sys

from setuptools import setup
from torch.utils import cpp_extension

filenames = [
    os.path.join(os.path.dirname(__file__), "fused_ops_frontend.cpp"),
    os.path.join(os.path.dirname(__file__), "multi_tensor_adam.cu"),
    os.path.join(os.path.dirname(__file__), "multi_tensor_scale_kernel.cu"),
    os.path.join(os.path.dirname(__file__), "multi_tensor_axpby_kernel.cu"),
    os.path.join(os.path.dirname(__file__), "multi_tensor_l2norm_kernel.cu"),
]

use_rocm = True if os.environ["ONNXRUNTIME_ROCM_VERSION"] else False
extra_compile_args = {"cxx": ["-O3"]}
if not use_rocm:
    extra_compile_args.update({"nvcc": os.environ["ONNXRUNTIME_CUDA_NVCC_EXTRA_ARGS"].split(",")})

setup(
    name="fused_ops",
    ext_modules=[
        cpp_extension.CUDAExtension(name="fused_ops", sources=filenames, extra_compile_args=extra_compile_args)
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
