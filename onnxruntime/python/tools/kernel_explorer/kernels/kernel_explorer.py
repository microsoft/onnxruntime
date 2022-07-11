# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Wrapper for native _kernel_explorer.so library"""

import ctypes
import os
import sys

build_dir = os.environ.get("KERNEL_EXPLORER_BUILD_DIR", None)
if build_dir is None:
    raise ValueError("Environment variable KERNEL_EXPLORER_BUILD_DIR is required")

if not os.path.exists(build_dir):
    raise ValueError(f"KERNEL_EXPLORER_BUILD_DIR ({build_dir}) points to nonexistent path")

build_dir = os.path.realpath(build_dir)
search_paths = [build_dir]

# As Kernel Explorer makes use of utility functions in ONNXRuntime, we dlopen all relevant libraries to bring required
# symbols into global namespace, so that we don't need to worry about linking.
library_files_to_load = [
    "onnxruntime_pybind11_state.so",
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_rocm.so",
]

library_to_load = []

for lib in library_files_to_load:
    for prefix in search_paths:
        path = os.path.join(prefix, lib)
        if os.path.exists(path):
            library_to_load.append(path)
            continue

        raise EnvironmentError(f"cannot found {lib}")


# onnxruntime_pybind11_state and kernel_explorer
sys.path.insert(0, build_dir)

# pylint: disable=wrong-import-position
import onnxruntime_pybind11_state  # noqa

# We need to call some functions to properly initialize so pointers in the library
onnxruntime_pybind11_state.get_available_providers()

# use RTLD_GLOBAL to bring all symbols to global name space
libraries = [ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL) for lib_path in library_to_load]

# pylint: disable=wrong-import-position, disable=unused-import
import _kernel_explorer  # noqa

# pylint: disable=wrong-import-position, disable=unused-import, disable=wildcard-import
from _kernel_explorer import *  # noqa
