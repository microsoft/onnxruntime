# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

from setuptools import setup
from torch.utils import cpp_extension

filename = os.path.join(os.path.dirname(__file__), "aten_op_executor/aten_op_executor.cc")

setup(
    name="ort_torch_ext",
    version="1.0",
    ext_modules=[cpp_extension.CppExtension(name="ort_torch_ext.aten_op_executor", sources=[filename])],
    packages=["ort_torch_ext"],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
