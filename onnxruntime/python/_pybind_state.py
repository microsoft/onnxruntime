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
        cuda_version_major, cuda_version_minor = version_info.cuda_version.split(".")
        if int(cuda_version_major) < 11:
            # Prior to CUDA 11 both major and minor version at build time/runtime have to match.
            cuda_env_variable = f"CUDA_PATH_V{cuda_version_major}_{cuda_version_minor}"
            if cuda_env_variable not in os.environ:
                raise ImportError(f"CUDA Toolkit {version_info.cuda_version} not installed on the machine.")
        else:
            # With CUDA 11 and newer only the major version at build time/runtime has to match.
            # Use the most recent minor version available.
            cuda_env_variable = None
            for i in range(9, -1, -1):
                if f"CUDA_PATH_V{cuda_version_major}_{i}" in os.environ:
                    cuda_env_variable = f"CUDA_PATH_V{cuda_version_major}_{i}"
                    break
            if not cuda_env_variable:
                raise ImportError(f"CUDA Toolkit {cuda_version_major}.x not installed on the machine.")

        cuda_bin_dir = os.path.join(os.environ[cuda_env_variable], "bin")
        if not os.path.isfile(os.path.join(cuda_bin_dir, f"cudnn64_{version_info.cudnn_version}.dll")):
            raise ImportError(f"cuDNN {version_info.cudnn_version} not installed in {cuda_bin_dir}.")

        if sys.version_info >= (3, 8):
            # Python 3.8 (and later) doesn't search system PATH when loading DLLs, so the CUDA location needs to be
            # specified explicitly using the new API introduced in Python 3.8.
            os.add_dll_directory(cuda_bin_dir)
            cuda_root = os.path.join(cuda_bin_dir, "..", "..")
            for root, _, files in os.walk(cuda_root):
                for f in files:
                    if f == "cupti.lib":
                        os.add_dll_directory(root)
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
