import threading
from functools import wraps

import torch  # noqa: F401

from onnxruntime.capi import _pybind_state as _C

from .aten_op_executor import execute_aten_operator_address, is_tensor_argument_address


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
def load_aten_op_executor_cpp_extension():
    _C.register_aten_op_executor(str(is_tensor_argument_address()), str(execute_aten_operator_address()))


def init_aten_op_executor():
    load_aten_op_executor_cpp_extension()
