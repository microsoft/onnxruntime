# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Initialize static objects needed to run custom autograd.Function's.
def enable_custom_autograd_support():
    from onnxruntime.capi._pybind_state import register_forward_runner, register_backward_runner
    from torch.onnx import register_custom_op_symbolic
    from ._custom_autograd_function_exporter import _export
    from ._custom_autograd_function_runner import call_python_forward_function, call_python_backward_function

    register_forward_runner(call_python_forward_function, False)
    register_backward_runner(call_python_backward_function, False)

    register_custom_op_symbolic('::prim_PythonOp', _export, 1)