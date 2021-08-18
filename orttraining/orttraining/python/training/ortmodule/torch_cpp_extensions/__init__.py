# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Support for PyTorch C++ extensions within ORTModule


TODO: Implement mechanism to register extensions and prevent issues with incorrect/missing flags
      for each :meth:`torch.utils.cpp_extension.*` call
"""

import os
import threading
from functools import wraps
from glob import glob
from onnxruntime.capi import _pybind_state as C


def is_installed(torch_cpp_extension_path):
    torch_cpp_exts = glob(os.path.join(torch_cpp_extension_path, '*.so'))
    torch_cpp_exts.extend(glob(os.path.join(torch_cpp_extension_path, '*.dll')))
    torch_cpp_exts.extend(glob(os.path.join(torch_cpp_extension_path, '*.dylib')))
    return len(torch_cpp_exts) > 0

def run_once_aten_op_executor(f):
    """
    Decorator to run a function only once.
    :param f: function to be run only once during execution time despite the number of calls
    :return: The original function with the params passed to it if it hasn't already been run before
    """
    @wraps(f)
    def aten_op_executor_wrapper(*args, **kwargs):
        if not aten_op_executor_wrapper.has_run:
            with aten_op_executor_wrapper.lock:
                if not aten_op_executor_wrapper.has_run:
                    aten_op_executor_wrapper.has_run = True
                    return f(*args, **kwargs)

    aten_op_executor_wrapper.lock = threading.Lock()
    aten_op_executor_wrapper.has_run = False
    return aten_op_executor_wrapper


@run_once_aten_op_executor
def _load_aten_op_executor_cpp_extension():
    from onnxruntime.training.ortmodule.torch_cpp_extensions import aten_op_executor
    C.register_aten_op_executor(str(aten_op_executor.is_tensor_argument_address()),
                                str(aten_op_executor.execute_aten_operator_address()))


def _load_aten_op_executor_cpp_extension_if_needed(onnx_model):
    for node in onnx_model.graph.node:
        if node.op_type == 'ATenOp' and node.domain == 'com.microsoft':
            _load_aten_op_executor_cpp_extension()
            break
