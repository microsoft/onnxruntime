#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
#--------------------------------------------------------------------------
import sys
import os
import warnings

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import *  # noqa
except ImportError as e:
    warnings.warn("Cannot load onnxruntime.capi. Error: '{0}'".format(str(e)))
