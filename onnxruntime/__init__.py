# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
ONNX Runtime is a performance-focused scoring engine for Open Neural Network Exchange (ONNX) models.
For more information on ONNX Runtime, please see `aka.ms/onnxruntime <https://aka.ms/onnxruntime/>`_
or the `Github project <https://github.com/microsoft/onnxruntime/>`_.
"""
__version__ = "1.4.0"
__author__ = "Microsoft"

from onnxruntime.capi._pybind_state import get_all_providers, get_available_providers, get_device, set_seed, \
    RunOptions, SessionOptions, set_default_logger_severity, NodeArg, ModelMetadata, GraphOptimizationLevel, \
    ExecutionMode, OrtDevice, SessionIOBinding

try:
    from onnxruntime.capi._pybind_state import set_cuda_mem_limit, set_cuda_device_id
except ImportError:
    pass

from onnxruntime.capi.session import InferenceSession, IOBinding
from onnxruntime.capi import onnxruntime_validation

from onnxruntime.capi.training import *  # noqa: F403

# TODO: thiagofc: Temporary experimental namespace for new PyTorch front-end
try:
    from . import experimental
except ImportError:
    pass

onnxruntime_validation.check_distro_info()
