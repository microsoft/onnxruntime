#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
"""
ONNX Runtime
enables high-performance evaluation of trained machine learning (ML)
models while keeping resource usage low.
Building on Microsoft's dedication to the
`Open Neural Network Exchange (ONNX) <https://onnx.ai/>`_
community, it supports traditional ML models as well
as Deep Learning algorithms in the
`ONNX-ML format <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_.
"""
__version__ = "0.4.0"
__author__ = "Microsoft"

from onnxruntime.capi import onnxruntime_validation
onnxruntime_validation.check_distro_info()
from onnxruntime.capi.session import InferenceSession
from onnxruntime.capi._pybind_state import RunOptions, SessionOptions, get_device, NodeArg, ModelMetadata
