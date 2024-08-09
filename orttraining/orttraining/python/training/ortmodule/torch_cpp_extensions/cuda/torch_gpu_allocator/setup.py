# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import fileinput
import os
import sys

from setuptools import setup
from torch.utils import cpp_extension

# TODO: Implement a cleaner way to auto-generate torch_gpu_allocator.cc
use_rocm = bool(os.environ["ONNXRUNTIME_ROCM_VERSION"])
gpu_identifier = "hip" if use_rocm else "cuda"
gpu_allocator_header = "HIPCachingAllocator" if use_rocm else "CUDACachingAllocator"
filename = os.path.join(os.path.dirname(__file__), "torch_gpu_allocator.cc")
with fileinput.FileInput(filename, inplace=True) as file:
    for line in file:
        if "___gpu_identifier___" in line:
            line = line.replace("___gpu_identifier___", gpu_identifier)  # noqa: PLW2901
        if "___gpu_allocator_header___" in line:
            line = line.replace("___gpu_allocator_header___", gpu_allocator_header)  # noqa: PLW2901
        sys.stdout.write(line)

extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}
if not use_rocm:
    nvcc_extra_args = os.environ.get("ONNXRUNTIME_CUDA_NVCC_EXTRA_ARGS", "")
    if nvcc_extra_args:
        extra_compile_args.update({"nvcc": nvcc_extra_args.split(",")})

setup(
    name="torch_gpu_allocator",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="torch_gpu_allocator", sources=[filename], extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
