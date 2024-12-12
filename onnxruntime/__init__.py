# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
ONNX Runtime is a performance-focused scoring engine for Open Neural Network Exchange (ONNX) models.
For more information on ONNX Runtime, please see `aka.ms/onnxruntime <https://aka.ms/onnxruntime/>`_
or the `Github project <https://github.com/microsoft/onnxruntime/>`_.
"""
__version__ = "1.21.0"
__author__ = "Microsoft"

# we need to do device version validation (for example to check Cuda version for an onnxruntime-training package).
# in order to know whether the onnxruntime package is for training it needs
# to do import onnxruntime.training.ortmodule first.
# onnxruntime.capi._pybind_state is required before import onnxruntime.training.ortmodule.
# however, import onnxruntime.capi._pybind_state will already raise an exception if a required Cuda version
# is not found.
# here we need to save the exception and continue with Cuda version validation in order to post
# meaningful messages to the user.
# the saved exception is raised after device version validation.
try:
    from onnxruntime.capi._pybind_state import ExecutionMode  # noqa: F401
    from onnxruntime.capi._pybind_state import ExecutionOrder  # noqa: F401
    from onnxruntime.capi._pybind_state import GraphOptimizationLevel  # noqa: F401
    from onnxruntime.capi._pybind_state import LoraAdapter  # noqa: F401
    from onnxruntime.capi._pybind_state import ModelMetadata  # noqa: F401
    from onnxruntime.capi._pybind_state import NodeArg  # noqa: F401
    from onnxruntime.capi._pybind_state import OrtAllocatorType  # noqa: F401
    from onnxruntime.capi._pybind_state import OrtArenaCfg  # noqa: F401
    from onnxruntime.capi._pybind_state import OrtMemoryInfo  # noqa: F401
    from onnxruntime.capi._pybind_state import OrtMemType  # noqa: F401
    from onnxruntime.capi._pybind_state import OrtSparseFormat  # noqa: F401
    from onnxruntime.capi._pybind_state import RunOptions  # noqa: F401
    from onnxruntime.capi._pybind_state import SessionIOBinding  # noqa: F401
    from onnxruntime.capi._pybind_state import SessionOptions  # noqa: F401
    from onnxruntime.capi._pybind_state import create_and_register_allocator  # noqa: F401
    from onnxruntime.capi._pybind_state import create_and_register_allocator_v2  # noqa: F401
    from onnxruntime.capi._pybind_state import disable_telemetry_events  # noqa: F401
    from onnxruntime.capi._pybind_state import enable_telemetry_events  # noqa: F401
    from onnxruntime.capi._pybind_state import get_all_providers  # noqa: F401
    from onnxruntime.capi._pybind_state import get_available_providers  # noqa: F401
    from onnxruntime.capi._pybind_state import get_build_info  # noqa: F401
    from onnxruntime.capi._pybind_state import get_device  # noqa: F401
    from onnxruntime.capi._pybind_state import get_version_string  # noqa: F401
    from onnxruntime.capi._pybind_state import has_collective_ops  # noqa: F401
    from onnxruntime.capi._pybind_state import set_default_logger_severity  # noqa: F401
    from onnxruntime.capi._pybind_state import set_default_logger_verbosity  # noqa: F401
    from onnxruntime.capi._pybind_state import set_seed  # noqa: F401

    import_capi_exception = None
except Exception as e:
    import_capi_exception = e

from onnxruntime.capi import onnxruntime_validation

if import_capi_exception:
    raise import_capi_exception

from onnxruntime.capi.onnxruntime_inference_collection import AdapterFormat  # noqa: F401
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession  # noqa: F401
from onnxruntime.capi.onnxruntime_inference_collection import IOBinding  # noqa: F401
from onnxruntime.capi.onnxruntime_inference_collection import OrtDevice  # noqa: F401
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue  # noqa: F401
from onnxruntime.capi.onnxruntime_inference_collection import SparseTensor  # noqa: F401

# TODO: thiagofc: Temporary experimental namespace for new PyTorch front-end
try:  # noqa: SIM105
    from . import experimental  # noqa: F401
except ImportError:
    pass

from onnxruntime.capi.onnxruntime_validation import cuda_version, package_name, version  # noqa: F401

if version:
    __version__ = version

onnxruntime_validation.check_distro_info()

# Load nvidia libraries from site-packages/nvidia if the package is onnxruntime-gpu
if (
    __package__ == "onnxruntime-gpu"
    # Just in case we rename the package name in the future
    or __package__ == "onnxruntime-cuda"
    or __package__ == "onnxruntime_gpu"
    or __package__ == "onnxruntime_cuda"
):
    import ctypes
    import os
    import platform
    import re
    import site
    import logging
    # Get the site-packages path where nvidia packages are installed
    site_packages_path = site.getsitepackages()[-1]
    nvidia_path = os.path.join(site_packages_path, "nvidia")
    # Traverse the directory and subdirectories
    if platform.system() == "Windows":  #
        # Define the list of DLL patterns, curand and nvJitLink are not included for Windows
        cuda_libs = (
            "cublas",
            "cublasLt",
            "cudnn",
            "cudart",
            "cufft",
            # "curand",
            # "nvJitLink",
        )
        # Construct a regex pattern for each library name with optional parts
        # Pattern explanation:
        # - `libname`: Match the base library name (e.g., "cudart")
        # - `(?:64)?`: Optionally match "64"
        # - `(?:_\d+)*`: Match zero or more occurrences of "_n" where "n" is one or more digits
        # - `.dll$`: End with ".dll" ignoring case
        lib_pattern = {lib: re.compile(rf"{lib}(?:64)?(?:_\d+)*\.dll$", re.IGNORECASE) for lib in cuda_libs}
        # Collect all directories under site-packages/nvidia that contain .dll files (for Windows)
        for root, _, files in os.walk(nvidia_path):
            # Add the current directory to the DLL search path

            with os.add_dll_directory(root):
                # Find all .dll files in the current directory
                for file in files:
                    for pattern in lib_pattern.items().values():
                        if pattern.match(file):
                            dll_path = os.path.join(root, file)
                            try:
                                _ = ctypes.CDLL(dll_path)
                            except Exception as e:
                                logging.error(f"Failed to load {dll_path}: {e}")
    elif platform.system() == "Linux":
        # Define the patterns with optional version number and case-insensitivity
        cuda_libs = (
            "libcublas.so",
            "libcublasLt.so",
            "libcudnn.so",
            "libcudart.so",
            "libnvrtc.so",
            "libcufft.so",
            "libcurand.so",
            "libnvJitLink.so",
        )

        # Regular expression to match .so files with optional versioning (e.g., .so, .so.1, .so.2.3)
        lib_pattern = {pattern: re.compile(rf"{re.escape(pattern)}(\.\d+)*$", re.IGNORECASE) for pattern in cuda_libs}

        # Traverse the directory and subdirectories
        for root, _, files in os.walk(nvidia_path):
            for file in files:
                # Check if the file matches the .so pattern
                for regex in lib_pattern.items().values():
                    if regex.match(file):  # Check if the file matches the pattern
                        so_path = os.path.join(root, file)
                        _ = ctypes.CDLL(so_path)
    else:
        pass
