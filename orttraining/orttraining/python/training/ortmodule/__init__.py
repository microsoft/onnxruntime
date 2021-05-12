# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from packaging import version

# All global constant goes here, before ORTModule is imported
ONNX_OPSET_VERSION = 12
MINIMUM_TORCH_VERSION_STR = '1.8.1'

from .ortmodule import ORTModule

# Verify proper PyTorch is installed before proceding to ONNX Runtime initializetion
try:
    import torch
    torch_version = version.parse(torch.__version__.split('+')[0])
    minimum_torch_version = version.parse(MINIMUM_TORCH_VERSION_STR)
    if torch_version < minimum_torch_version:
        raise RuntimeError(
            f'ONNXRuntime ORTModule frontend requires PyTorch version greater or equal to {MINIMUM_TORCH_VERSION_STR}, '
            f'but version {torch.__version__} was found instead.')

    import onnxruntime
    from ._custom_autograd_function_runner import call_python_forward_function, call_python_backward_function
    onnxruntime.register_forward_runner(call_python_forward_function)
    onnxruntime.register_backward_runner(call_python_backward_function)

    from torch.onnx import register_custom_op_symbolic
    from ._custom_autograd_function_exporter import custom_autograd_function_exporter
    register_custom_op_symbolic('::prim_PythonOp', custom_autograd_function_exporter, 1)

except:
    raise(f'PyTorch {MINIMUM_TORCH_VERSION_STR} must be installed in order to run ONNXRuntime ORTModule frontend!')
