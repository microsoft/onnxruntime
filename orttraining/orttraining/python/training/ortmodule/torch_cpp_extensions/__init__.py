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

TODO: Create a generic mechanism to pass arguments from ORTModule into each extension setup.py
TODO: Create environment variables to allow extensions to be hosted outside ONNX runtime installation folder
      (e.g. ORTMODULE_EXTERNAL_TORCH_CPP_EXTENSION_DIR, ORTMODULE_EXTERNAL_TORCH_CUDA_EXTENSION_DIR)

"""
import warnings


def get_fused_ops_extension():
    try:
        from onnxruntime.training.ortmodule.torch_cpp_extensions import fused_ops
    except ImportError:
        warnings.warn("Imported fused_ops from site-packages.", ImportWarning)
        import fused_ops
    return fused_ops


def get_aten_op_executor_extension():
    try:
        from onnxruntime.training.ortmodule.torch_cpp_extensions import aten_op_executor
    except ImportError:
        warnings.warn("Imported aten_op_executor from site-packages.", ImportWarning)
        import aten_op_executor
    return aten_op_executor


def get_torch_interop_utils_extension():
    try:
        from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_interop_utils
    except ImportError:
        warnings.warn("Imported torch_interop_utils from site-packages.", ImportWarning)
        import torch_interop_utils
    return torch_interop_utils


def get_torch_gpu_allocator_extension():
    try:
        from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_gpu_allocator
    except ImportError:
        warnings.warn("Imported torch_gpu_allocator from site-packages.", ImportWarning)
        import torch_gpu_allocator
    return torch_gpu_allocator


def is_installed(torch_cpp_extension_path):
    try:
        get_aten_op_executor_extension()
    except ImportError:
        return False
    try:
        get_torch_interop_utils_extension()
    except ImportError:
        return False
    return True
