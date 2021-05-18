# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from packaging import version


# All global constant goes here, before ORTModule is imported
ONNX_OPSET_VERSION = 12
MINIMUM_TORCH_VERSION_STR = '1.8.1'

# Check whether Torch C++ extension compilation was aborted in previous runs
TORCH_CPP_BUILD_DIR = os.path.join(os.path.dirname(__file__),'torch_inline_extensions')
if not os.path.exists(TORCH_CPP_BUILD_DIR):
    os.makedirs(TORCH_CPP_BUILD_DIR, exist_ok = True)
elif os.path.exists(os.path.join(TORCH_CPP_BUILD_DIR,'lock')):
    print("WARNING: ORTModule detected PyTorch CPP extension's lock file during initialization, "
          "which can cause unexpected hangs. "
          f"Delete {os.path.join(TORCH_CPP_BUILD_DIR,'lock')} to supress this warning.")

# Verify proper PyTorch is installed before proceding to ONNX Runtime initializetion
try:
    import torch
    torch_version = version.parse(torch.__version__.split('+')[0])
    minimum_torch_version = version.parse(MINIMUM_TORCH_VERSION_STR)
    if torch_version < minimum_torch_version:
        raise RuntimeError(
            f'ONNXRuntime ORTModule frontend requires PyTorch version greater or equal to {MINIMUM_TORCH_VERSION_STR}, '
            f'but version {torch.__version__} was found instead.')
except:
    raise(f'PyTorch {MINIMUM_TORCH_VERSION_STR} must be installed in order to run ONNXRuntime ORTModule frontend!')

# ORTModule must be loaded only after all validation passes
from .ortmodule import ORTModule
