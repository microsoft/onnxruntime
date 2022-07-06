# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

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

# kernel explorer hooks up some underlaying functions for microbenchmark purpose. Some utility functions might also be
# needed. We dlopen all this libraries to bring all those required symbols into global namespace so that we don't need
# worry about linking.
library_files_to_load = [
    "onnxruntime_pybind11_state.so",
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_rocm.so"
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

import onnxruntime_pybind11_state

# We need to call some functions to properly initialize so pointers in the library
onnxruntime_pybind11_state.get_available_providers()

# use RTLD_GLOBAL to bring all symbols to global name space
libraries = [ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL) for lib_path in library_to_load]

import _kernel_explorer
from _kernel_explorer import *
