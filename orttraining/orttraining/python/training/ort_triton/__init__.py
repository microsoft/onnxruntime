# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import threading
from functools import wraps

from onnxruntime.capi import _pybind_state as _C

from .kernel import *  # noqa: F403
from .triton_op_executor import call_triton_by_name, call_triton_by_onnx, get_config


def run_once_register_triton_op_executor(f):
    """
    Decorator to run a function only once.
    :param f: function to be run only once during execution time despite the number of calls
    :return: The original function with the params passed to it if it hasn't already been run before
    """

    @wraps(f)
    def register_triton_op_executor_wrapper(*args, **kwargs):
        if not register_triton_op_executor_wrapper.has_run:
            with register_triton_op_executor_wrapper.lock:
                if not register_triton_op_executor_wrapper.has_run:
                    register_triton_op_executor_wrapper.has_run = True
                    f(*args, **kwargs)

    register_triton_op_executor_wrapper.lock = threading.Lock()
    register_triton_op_executor_wrapper.has_run = False
    return register_triton_op_executor_wrapper


@run_once_register_triton_op_executor
def register_triton_op_executor():
    _C.register_triton_op_executor(get_config, call_triton_by_name, call_triton_by_onnx)
