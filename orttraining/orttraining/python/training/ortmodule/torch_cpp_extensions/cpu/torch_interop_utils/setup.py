# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

from setuptools import Extension, setup  # noqa: F401
from torch.utils import cpp_extension

source_filenames = [
    "torch_interop_utils.cc",
    "ctx_pool.cc",
    "custom_function_bw.cc",
    "custom_function_fw.cc",
    "custom_function_shared.cc",
]

cur_file_dir = os.path.dirname(__file__)

header_filenames = [
    # "/usr/local/cuda/include/", # uncomment this line to build nvtx support,
    cur_file_dir,
]

extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}
setup(
    name="torch_interop_utils",
    ext_modules=[
        cpp_extension.CppExtension(
            name="torch_interop_utils",
            sources=[os.path.join(cur_file_dir, filename) for filename in source_filenames],
            extra_compile_args=extra_compile_args,
            include_dirs=header_filenames,
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
