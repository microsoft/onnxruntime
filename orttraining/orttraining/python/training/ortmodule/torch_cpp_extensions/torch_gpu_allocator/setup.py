# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import fileinput
from setuptools import setup, Extension
import sys
from torch.utils import cpp_extension


def parse_arg_remove_boolean(argv, arg_name):
    arg_value = False
    if arg_name in sys.argv:
        arg_value = True
        argv.remove(arg_name)

    return arg_value

use_rocm = True if parse_arg_remove_boolean(sys.argv, '--use_rocm') else False
gpu_identifier = "hip" if use_rocm else "cuda"
gpu_allocator_header = "HIPCachingAllocator" if use_rocm else "CUDACachingAllocator"

with fileinput.FileInput('torch_gpu_allocator.cpp.template', inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace('___gpu_identifier___', gpu_identifier), end='')
        print(line.replace('___gpu_allocator_header___', gpu_allocator_header), end='')

setup(name='torch_gpu_allocator',
      ext_modules=[cpp_extension.CUDAExtension('torch_gpu_allocator', ['torch_gpu_allocator.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
