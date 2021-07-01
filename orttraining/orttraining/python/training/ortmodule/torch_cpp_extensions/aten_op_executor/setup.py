# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

filename = os.path.join(os.path.dirname(__file__),
                        'aten_op_executor.cc')
setup(name='aten_op_executor',
      ext_modules=[cpp_extension.CppExtension(name='aten_op_executor',
                                              sources=[filename])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
