# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from setuptools import setup, Extension
from torch.utils import cpp_extension

filename = 'torch_cpp_extensions/aten_op_executor/aten_op_executor.cc'
setup(name='aten_op_executor',
      ext_modules=[cpp_extension.CppExtension('aten_op_executor', [filename])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
