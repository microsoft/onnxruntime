# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from packaging import version

################################################################################
# All global constant goes here, before ORTModule is imported ##################
################################################################################
ONNX_OPSET_VERSION = 12
MINIMUM_TORCH_VERSION_STR = '1.8.1'
TORCH_CPP_BUILD_DIR = os.path.join(os.path.dirname(__file__),'torch_inline_extensions')

# Check whether Torch C++ extension compilation was aborted in previous runs
if not os.path.exists(TORCH_CPP_BUILD_DIR):
    os.makedirs(TORCH_CPP_BUILD_DIR, exist_ok = True)
elif os.path.exists(os.path.join(TORCH_CPP_BUILD_DIR,'lock')):
    print("WARNING: ORTModule detected PyTorch CPP extension's lock file during initialization, "
          "which can cause unexpected hangs. "
          f"Delete {os.path.join(TORCH_CPP_BUILD_DIR,'lock')} to prevent unexpected behavior.")

# Verify proper PyTorch is installed before proceding to ONNX Runtime initializetion
try:
    import torch
    torch_version = version.parse(torch.__version__.split('+')[0])
    minimum_torch_version = version.parse(MINIMUM_TORCH_VERSION_STR)
    if torch_version < minimum_torch_version:
        raise RuntimeError(
            f'ONNXRuntime ORTModule frontend requires PyTorch version greater or equal to {MINIMUM_TORCH_VERSION_STR}, '
            f'but version {torch.__version__} was found instead.')

    from onnxruntime.capi._pybind_state import register_forward_runner, register_backward_runner
    from ._custom_autograd_function_runner import call_python_forward_function, call_python_backward_function
    register_forward_runner(call_python_forward_function, False)
    register_backward_runner(call_python_backward_function, False)

    from torch.onnx import register_custom_op_symbolic
    from ._custom_autograd_function_exporter import _export
    register_custom_op_symbolic('::prim_PythonOp', _export, 1)

except:
    raise(f'PyTorch {MINIMUM_TORCH_VERSION_STR} must be installed in order to run ONNXRuntime ORTModule frontend!')

# ORTModule must be loaded only after all validation passes
from .ortmodule import ORTModule
