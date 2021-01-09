# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
ONNX Runtime is a performance-focused scoring engine for Open Neural Network Exchange (ONNX) models.
For more information on ONNX Runtime, please see `aka.ms/onnxruntime <https://aka.ms/onnxruntime/>`_
or the `Github project <https://github.com/microsoft/onnxruntime/>`_.
"""
__version__ = "1.6.0"
__author__ = "Microsoft"

import os
import platform
import sys

# Python 3.8 (and later) on Windows doesn't search system PATH when loading DLLs,
# so CUDA location needs to be specified explicitly. This needs to be done before importing
# onnxruntime.capi._pybind_state
if "CUDA_PATH" in os.environ and platform.system() == "Windows" and sys.version_info >= (3, 8):
    cuda_bin_dir = os.path.join(os.environ["CUDA_PATH"], "bin")
    os.add_dll_directory(cuda_bin_dir)

from onnxruntime.capi._pybind_state import get_all_providers, get_available_providers, get_device, set_seed, \
    RunOptions, SessionOptions, set_default_logger_severity, enable_telemetry_events, disable_telemetry_events, \
    NodeArg, ModelMetadata, GraphOptimizationLevel, ExecutionMode, ExecutionOrder, OrtDevice, SessionIOBinding, \
    OrtAllocatorType, OrtMemType, OrtArenaCfg, OrtMemoryInfo, create_and_register_allocator

from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession, IOBinding, OrtValue
from onnxruntime.capi import onnxruntime_validation

from onnxruntime.capi.training import *  # noqa: F403

# TODO: thiagofc: Temporary experimental namespace for new PyTorch front-end
try:
    from . import experimental
except ImportError:
    pass

onnxruntime_validation.check_distro_info()
