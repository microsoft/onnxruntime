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
use_rocm = True if os.environ["ONNXRUNTIME_ROCM_VERSION"] else False
gpu_identifier = "hip" if use_rocm else "cuda"
gpu_allocator_header = "HIPCachingAllocator" if use_rocm else "CUDACachingAllocator"
filename = os.path.join(os.path.dirname(__file__), "torch_gpu_allocator.cc")
with fileinput.FileInput(filename, inplace=True) as file:
    for line in file:
        if "___gpu_identifier___" in line:
            line = line.replace("___gpu_identifier___", gpu_identifier)
        if "___gpu_allocator_header___" in line:
            line = line.replace("___gpu_allocator_header___", gpu_allocator_header)
        sys.stdout.write(line)

extra_compile_args = {"cxx": ["-O3"]}
if not use_rocm:
    extra_compile_args.update({"nvcc": os.environ["ONNXRUNTIME_CUDA_NVCC_EXTRA_ARGS"].split(",")})

setup(
    name="torch_gpu_allocator",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="torch_gpu_allocator", sources=[filename], extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
