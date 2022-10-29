# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Support for PyTorch C++ extensions within ORTModule

Pytorch C++ extensions must be added to this (ORTMODULE_TORCH_CPP_DIR) folder to be automatically
detected and installed by `python -m torch_ort.configure`

Each extension must be within a folder and contain a setup.py file

CUDA extensions must be stored within 'cuda' folders
CPU extensions must be stored within 'cpu' subfolder

Extensions are lexicographically ordered for compilation.
e.g. '001_my_extension' is compiled before '002_my_other_extension'

The following environment variables are available for the extensions setup.py

    - ORTMODULE_TORCH_CPP_DIR: ORTModule's internal
    - ONNXRUNTIME_ROCM_VERSION: ROCM version used to build ONNX Runtime package
    - ONNXRUNTIME_CUDA_VERSION: CUDA version used to build ONNX Runtime package
    - ONNXRUNTIME_FORCE_CUDA: Force CUDA extensions to be used when it is not available to build ONNX Runtime package

TODO: Create a generic mechanism to pass arguments from ORTModule into each extension setup.py
TODO: Create environment variables to allow extensions to be hosted outside ONNX runtime installation folder
      (e.g. ORTMODULE_EXTERNAL_TORCH_CPP_EXTENSION_DIR, ORTMODULE_EXTERNAL_TORCH_CUDA_EXTENSION_DIR)

"""

import os
from glob import glob


def is_installed(torch_cpp_extension_path):
    torch_cpp_exts = glob(os.path.join(torch_cpp_extension_path, "*.so"))
    torch_cpp_exts.extend(glob(os.path.join(torch_cpp_extension_path, "*.dll")))
    torch_cpp_exts.extend(glob(os.path.join(torch_cpp_extension_path, "*.dylib")))
    return len(torch_cpp_exts) > 0
