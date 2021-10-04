# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import fileinput
import os
import sys

from setuptools import setup
from torch.utils import cpp_extension

filenames = [os.path.join(os.path.dirname(__file__), 'fused_adam_frontend.cpp'),
             os.path.join(os.path.dirname(__file__), 'multi_tensor_adam.cu')]

setup(name='adam_optimizer',
      ext_modules=[cpp_extension.CUDAExtension(name='adam_optimizer',
                                              sources=filenames,
                                              extra_compile_args={
                                                'cxx': ['-O3', '-std=c++14', '-g', '-Wno-reorder'],
                                                'nvcc': ['-lineinfo', '-O3', '--use_fast_math']
                             })],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
