# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

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


class CustomAutogradFunctionEnabler(object):
    """Used to enable custom autograd function fallback feature. """
    def __init__(self):
        self._is_enabled = False
        self._callback = enable_custom_autograd_support

    @property
    def enable_state(self):
        return self._is_enabled

    @enable_state.setter
    def enable_state(self, new_val):
        """Once state updated to True, autograd function runner and custom exporters are registered through callback.
           This implies that enabling MUST be done before the ONNX model export.
        """
        self._is_enabled = new_val
        if self._is_enabled:
            self._callback()
