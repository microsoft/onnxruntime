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

# Initialize static objects needed to run custom autograd.Function's.
def enable_custom_autograd_support():
    from onnxruntime.capi._pybind_state import register_forward_runner, register_backward_runner, unregister_python_functions
    from torch.onnx import register_custom_op_symbolic
    from ._custom_autograd_function_exporter import _export
    from ._custom_autograd_function_runner import call_python_forward_function, call_python_backward_function
    import atexit

    register_forward_runner(call_python_forward_function)
    register_backward_runner(call_python_backward_function)

    # Unregister all python functions automatically upon normal interpreter termination.
    atexit.register(unregister_python_functions)

    register_custom_op_symbolic('::prim_PythonOp', _export, 1)

    custom_autograd_function_enabler.state = True
