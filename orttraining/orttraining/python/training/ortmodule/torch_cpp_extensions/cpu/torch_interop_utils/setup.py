# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

from setuptools import Extension, setup
from torch.utils import cpp_extension

filename = os.path.join(os.path.dirname(__file__), "torch_interop_utils.cc")
extra_compile_args = {"cxx": ["-O3"]}
setup(
    name="torch_interop_utils",
    ext_modules=[
        cpp_extension.CppExtension(
            name="torch_interop_utils", sources=[filename], extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
