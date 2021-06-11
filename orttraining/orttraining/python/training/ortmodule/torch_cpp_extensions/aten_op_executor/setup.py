# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='aten_op_executor',
      ext_modules=[cpp_extension.CUDAExtension('aten_op_executor', ['aten_op_executor.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
