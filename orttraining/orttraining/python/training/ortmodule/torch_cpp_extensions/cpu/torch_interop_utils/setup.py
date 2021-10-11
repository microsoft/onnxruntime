# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

filename = os.path.join(os.path.dirname(__file__),
                        'torch_interop_utils.cc')
setup(name='torch_interop_utils',
      ext_modules=[cpp_extension.CppExtension(name='torch_interop_utils',
                                              sources=[filename])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
