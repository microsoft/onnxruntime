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


def check_and_load_cuda_libs(root_directory, cuda_libs):
    # Convert the target library names to lowercase for case-insensitive comparison
    found_libs = {}
    for dirpath, _, filenames in os.walk(root_directory):
        # Convert filenames in the current directory to lowercase for comparison
        files_in_dir = {file.lower(): file for file in filenames}  # Map lowercase to original
        # Find common libraries in the current directory
        matched_libs = cuda_libs.intersection(files_in_dir.keys())
        for lib in matched_libs:
            # Store the full path of the found DLL
            full_path = os.path.join(dirpath, files_in_dir[lib])
            found_libs[lib] = full_path
            try:
                # Load the DLL using ctypes
                _ = ctypes.CDLL(full_path)
                logging.info(f"Successfully loaded: {full_path}")
            except OSError as e:
                logging.error(f"Failed to load {full_path}: {e}")

        # If all required libraries are found, stop the search
        if set(found_libs.keys()) == cuda_libs:
            print("All required CUDA libraries found and loaded.")
            return True
    logging.error(f"Failed to load all required CUDA libraries. missing libraries: {cuda_libs - found_libs.keys()}")
    return False


# Load nvidia libraries from site-packages/nvidia if the package is onnxruntime-gpu
if (
    __package__ == "onnxruntime-gpu"
    # Just in case we rename the package name in the future
    or __package__ == "onnxruntime-cuda"
    or __package__ == "onnxruntime_gpu"
    or __package__ == "onnxruntime_cuda"
):
    import ctypes
    import logging
    import os
    import platform
    import re
    import site

    # Get the site-packages path where nvidia packages are installed
    site_packages_path = site.getsitepackages()[-1]
    nvidia_path = os.path.join(site_packages_path, "nvidia")
    # Traverse the directory and subdirectories
    cuda_libs = ()
    if platform.system() == "Windows":  #
        # Define the list of DLL patterns, nvrtc, curand and nvJitLink are not included for Windows
        if (11, 0) <= cuda_version() < (12, 0):
            cuda_libs = (
                "cublasLT64_11.dll",
                "cublas64_11.dll",
                "cufft64_10.dll",
                "cudart64_11.dll",
                "cudnn64_8.dll",
            )
        elif (12, 0) <= cuda_version() < (13, 0):
            cuda_libs = (
                "cublasLT64_12.dll",
                "cublas64_12.dll",
                "cufft64_11.dll",
                "cudart64_12.dll",
                "cudnn64_9.dll",
            )
    elif platform.system() == "Linux":
        if (11, 0) <= cuda_version() < (12, 0):
            # Define the patterns with optional version number and case-insensitivity
            cuda_libs = (
                "libcublaslt.so.11",
                "libcublas.so.11",
                "libcurand.so.10",
                "libcufft.so.10",
                "libcudart.so.11",
                "libcudnn.so.8",
                "libnvrtc.so.11",
            )
        elif (12, 0) <= cuda_version() < (13, 0):
            cuda_libs = (
                "libcublaslt.so.12",
                "libcublas.so.12",
                "libcurand.so.10",
                "libcufft.so.11",
                "libcudart.so.12",
                "libcudnn.so.9",
                "libnvrtc.so.12",
            )
    else:
        logging.error(f"Unsupported platform: {platform.system()}")

    if cuda_libs:
        # Convert the target library names to lowercase for case-insensitive comparison
        cuda_libs = {lib.lower() for lib in cuda_libs}
        # Load the required CUDA libraries
        check_and_load_cuda_libs(nvidia_path, cuda_libs)
