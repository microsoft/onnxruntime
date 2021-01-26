# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Ensure that dependencies are available and then load the extension module.
"""
import os
import platform
import sys

from . import _ld_preload  # noqa: F401

if platform.system() == "Windows":
    from . import version_info

    if version_info.use_cuda:
        cuda_env_variable = "CUDA_PATH_V" + version_info.cuda_version.replace(".", "_")
        if cuda_env_variable not in os.environ:
            raise ImportError(f"CUDA Toolkit {version_info.cuda_version} not installed on the machine.")
        cuda_bin_dir = os.path.join(os.environ[cuda_env_variable], "bin")
        if not os.path.isfile(os.path.join(cuda_bin_dir, f"cudnn64_{version_info.cudnn_version}.dll")):
            raise ImportError("cuDNN {version_info.cudnn_version} not installed on the machine.")

        if sys.version_info >= (3, 8):
            # Python 3.8 (and later) doesn't search system PATH when loading DLLs, so the CUDA location needs to be
            # specified explicitly using the new API introduced in Python 3.8.
            os.add_dll_directory(cuda_bin_dir)
        else:
            # Python 3.7 (and earlier) searches directories listed in PATH variable.
            # Make sure that the target CUDA version is at the beginning (important if multiple CUDA versions are
            # installed on the machine.)
            os.environ["PATH"] += cuda_bin_dir + os.pathsep + os.environ["PATH"]

    if version_info.vs2019 and platform.architecture()[0] == "64bit":
        if not os.path.isfile("C:\\Windows\\System32\\vcruntime140_1.dll"):
            raise ImportError(
                "Microsoft Visual C++ Redistributable for Visual Studio 2019 not installed on the machine.")

from .onnxruntime_pybind11_state import *  # noqa
