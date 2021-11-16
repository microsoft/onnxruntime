# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import fileinput
import os
import sys

from setuptools import setup
from torch.utils import cpp_extension

def workaround_strict_prototypes_warning():
    # Calling this function to eliminate -Wstrict-prototypes warnings.
    # Per https://stackoverflow.com/a/29634231/23845, it's a 15+ year bug
    # https://bugs.python.org/issue1222585, which hasn't been fixed.
    # Following Pytorch, we use a workaround from stackoverflow.
    # This is safe because we only compile C++ code in this extension.
    import distutils.sysconfig
    cfg_vars = distutils.sysconfig.get_config_vars()
    for key, value in cfg_vars.items():
        if type(value) == str:
            cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

workaround_strict_prototypes_warning()

# TODO: Implement a cleaner way to auto-generate torch_gpu_allocator.cc
use_rocm = True if os.environ['ONNXRUNTIME_ROCM_VERSION'] else False
gpu_identifier = "hip" if use_rocm else "cuda"
gpu_allocator_header = "HIPCachingAllocator" if use_rocm else "CUDACachingAllocator"
filename = os.path.join(os.path.dirname(__file__),
                        'torch_gpu_allocator.cc')
with fileinput.FileInput(filename, inplace=True) as file:
    for line in file:
        if '___gpu_identifier___' in line:
            line = line.replace('___gpu_identifier___', gpu_identifier)
        if '___gpu_allocator_header___' in line:
            line = line.replace('___gpu_allocator_header___', gpu_allocator_header)
        sys.stdout.write(line)

setup(name='torch_gpu_allocator',
      ext_modules=[cpp_extension.CUDAExtension(name='torch_gpu_allocator',
                                               sources=[filename])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
