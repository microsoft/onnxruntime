# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Ensure that dependencies are available and then load the extension module.
"""
import os
import platform

from . import _ld_preload  # noqa: F401

if platform.system() == "Windows":
    from . import version_info

    if version_info.vs2019 and platform.architecture()[0] == "64bit":
        if not os.path.isfile("C:\\Windows\\System32\\vcruntime140_1.dll"):
            raise ImportError(
                "Microsoft Visual C++ Redistributable for Visual Studio 2019 not installed on the machine.")

from .onnxruntime_pybind11_state import *  # noqa
