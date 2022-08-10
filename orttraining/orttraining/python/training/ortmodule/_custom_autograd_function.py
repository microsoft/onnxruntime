# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


class Enabler(object):
    def __init__(self):
        self._state = False

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state = val


custom_autograd_function_enabler = Enabler()


def enable_custom_autograd_support():
    # Initialize static objects needed to run custom autograd.Function's.

    import atexit

    from torch.onnx import register_custom_op_symbolic

    from onnxruntime.capi._pybind_state import (
        register_backward_runner,
        register_forward_runner,
        unregister_python_functions,
    )
    from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_interop_utils

    from ._custom_autograd_function_exporter import _export
    from ._custom_autograd_function_runner import call_python_backward_function, call_python_forward_function

    register_forward_runner(call_python_forward_function)
    register_backward_runner(call_python_backward_function)

    # Unregister all python functions automatically upon normal interpreter termination.
    atexit.register(unregister_python_functions)
    # Clear all gradient functions, to avoid a deadlock issue.
    # Check the called function for more detailed comments.
    atexit.register(torch_interop_utils.clear_all_grad_fns)

    try:
        # This is for the latest Pytorch nightly after this commit:
        # https://github.com/pytorch/pytorch/commit/11bc435622e6b7207bbf37ed1aafe999e1f296ec
        register_custom_op_symbolic("prim::PythonOp", _export, 1)
    except:
        # This applies to Pytorch 1.9 and 1.9.1.
        register_custom_op_symbolic("::prim_PythonOp", _export, 1)

    custom_autograd_function_enabler.state = True


def toggle_custom_autograd_support(flag):
    if flag is True:
        enable_custom_autograd_support()


# Be noted, setup.py will replace below with "toggle_custom_autograd_support(True|False)" at the end of the file if
# enable_training_torch_interop is toggled during build.

toggle_custom_autograd_support(False)
