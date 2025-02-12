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
    from onnxruntime.capi._pybind_state import (
        ExecutionMode,  # noqa: F401
        ExecutionOrder,  # noqa: F401
        GraphOptimizationLevel,  # noqa: F401
        LoraAdapter,  # noqa: F401
        ModelMetadata,  # noqa: F401
        NodeArg,  # noqa: F401
        OrtAllocatorType,  # noqa: F401
        OrtArenaCfg,  # noqa: F401
        OrtMemoryInfo,  # noqa: F401
        OrtMemType,  # noqa: F401
        OrtSparseFormat,  # noqa: F401
        RunOptions,  # noqa: F401
        SessionIOBinding,  # noqa: F401
        SessionOptions,  # noqa: F401
        create_and_register_allocator,  # noqa: F401
        create_and_register_allocator_v2,  # noqa: F401
        disable_telemetry_events,  # noqa: F401
        enable_telemetry_events,  # noqa: F401
        get_all_providers,  # noqa: F401
        get_available_providers,  # noqa: F401
        get_build_info,  # noqa: F401
        get_device,  # noqa: F401
        get_version_string,  # noqa: F401
        has_collective_ops,  # noqa: F401
        set_default_logger_severity,  # noqa: F401
        set_default_logger_verbosity,  # noqa: F401
        set_seed,  # noqa: F401
    )

    import_capi_exception = None
except Exception as e:
    import_capi_exception = e

from onnxruntime.capi import onnxruntime_validation

if import_capi_exception:
    raise import_capi_exception

from onnxruntime.capi.onnxruntime_inference_collection import (
    AdapterFormat,  # noqa: F401
    InferenceSession,  # noqa: F401
    IOBinding,  # noqa: F401
    OrtDevice,  # noqa: F401
    OrtValue,  # noqa: F401
    SparseTensor,  # noqa: F401
)

# TODO: thiagofc: Temporary experimental namespace for new PyTorch front-end
try:  # noqa: SIM105
    from . import experimental  # noqa: F401
except ImportError:
    pass


def try_load_dlls_from_nvidia_packages(cuda_version):
    if not (cuda_version and cuda_version.startswith("12.")):
        return

    import ctypes
    import os
    import platform
    import site

    if platform.system() not in ["Windows", "Linux"]:
        return

    # Get the site-packages path where nvidia packages are installed
    site_packages_path = site.getsitepackages()[-1]
    nvidia_path = os.path.join(site_packages_path, "nvidia")
    if not os.path.exists(nvidia_path):
        return

    dll_paths = ()  # Path relative to the nvidia root.
    if platform.system() == "Windows":
        dll_paths = [
            ("cublas", "bin", "cublasLt64_12.dll"),
            ("cublas", "bin", "cublas64_12.dll"),
            ("cufft", "bin", "cufft64_11.dll"),
            ("cuda_runtime", "bin", "cudart64_12.dll"),
            ("cudnn", "bin", "cudnn64_9.dll"),
        ]
    else:  # platform.system() == "Linux".
        dll_paths = [
            ("cublas", "lib", "libcublas.so.12"),
            ("cublas", "lib", "libcublasLt.so.12"),
            ("cuda_nvrtc", "lib", "libnvrtc.so.12"),
            ("curand", "lib", "libcurand.so.10"),
            ("cufft", "lib", "libcufft.so.11"),
            ("cuda_runtime", "lib", "libcudart.so.12"),
            ("cudnn", "lib", "libcudnn.so.9"),
        ]

    for relative_path in dll_paths:
        full_dll_path = os.path.join(nvidia_path, *relative_path)
        if os.path.exists(full_dll_path):
            try:
                # Load the DLL using ctypes
                _ = ctypes.CDLL(full_dll_path)
            except Exception as e:
                print(f"Failed to load {full_dll_path}: {e}")


package_name, version, cuda_version = onnxruntime_validation.get_package_name_and_version_info()
try_load_dlls_from_nvidia_packages(cuda_version)

if version:
    __version__ = version

onnxruntime_validation.check_distro_info()
