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
import os

__version__ = ''
cwd = os.path.dirname(os.path.realpath(__file__))
with open(cwd + '/../VERSION_NUMBER') as f:
    __version__ = f.readline().strip() 
__author__ = "Microsoft"

from onnxruntime.capi import onnxruntime_validation
onnxruntime_validation.check_distro_info()
from onnxruntime.capi.session import InferenceSession
from onnxruntime.capi._pybind_state import RunOptions, SessionOptions, get_device, NodeArg, ModelMetadata
