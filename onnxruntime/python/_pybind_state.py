# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import platform
import sys
import warnings
import onnxruntime.capi._ld_preload  # noqa: F401

# Python 3.8 (and later) on Windows doesn't search system PATH when loading DLLs,
# so CUDA location needs to be specified explicitly.
if platform.system() == "Windows" and sys.version_info >= (3, 8):
    CUDA_VERSION = "10.2"
    CUDNN_VERSION = "8"
    cuda_env_variable = "CUDA_PATH_V" + CUDA_VERSION.replace(".", "_")
    if cuda_env_variable not in os.environ:
        raise ImportError(f"CUDA Toolkit {CUDA_VERSION} not installed on the machine.")
    cuda_bin_dir = os.path.join(os.environ[cuda_env_variable], "bin")
    if not os.path.isfile(os.path.join(cuda_bin_dir, f"cudnn64_{CUDNN_VERSION}.dll")):
        raise ImportError(f"cuDNN {CUDNN_VERSION} not installed on the machine.")
    os.add_dll_directory(cuda_bin_dir)

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import *  # noqa
except ImportError as e:
    warnings.warn("Cannot load onnxruntime.capi. Error: '{0}'.".format(str(e)))

    # If on Windows, check if this import error is caused by the user not installing the 2019 VC Runtime
    # The VC Redist installer usually puts the VC Runtime dlls in the System32 folder
    # This may not always paint the true picture as anyone building from source using VS 2017 might hit this error
    # because the machine might be missing the 2019 VC Runtime but it is not actually needed in that case and the
    # import error might actually be due to some other reason.
    # TODO: Add a guard against False Positive error message
    # As a proxy for checking if the 2019 VC Runtime is installed,
    # we look for a specific dll only shipped with the 2019 VC Runtime
    if platform.system() == "Windows" and not os.path.isfile("C:\\Windows\\System32\\vcruntime140_1.dll"):
        warnings.warn("Unless you have built the wheel using VS 2017, "
                      "please install the 2019 Visual C++ runtime and then try again.")
