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

filename = 'torch_cpp_extensions/torch_gpu_allocator/torch_gpu_allocator.cc'
with fileinput.FileInput(filename, inplace=True) as file:
    for line in file:
        if '___gpu_identifier___' in line:
            line = line.replace('___gpu_identifier___', gpu_identifier)
        if '___gpu_allocator_header___' in line:
            line = line.replace('___gpu_allocator_header___', gpu_allocator_header)
        sys.stdout.write(line)

setup(name='torch_gpu_allocator',
      ext_modules=[cpp_extension.CUDAExtension('torch_gpu_allocator', [filename])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
