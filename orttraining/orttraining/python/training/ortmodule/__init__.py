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

# Check Torch CPP extension build directory
home_dir = os.path.expanduser("~")
python_package_dir = os.path.dirname(__file__)
TORCH_CPP_BUILD_DIR = os.path.join(python_package_dir,'torch_inline_extensions')
TORCH_CPP_BUILD_DIR_BACKUP = os.path.join(home_dir, '.cache', 'torch_ort_extensions')
if not os.access(python_package_dir, os.X_OK | os.W_OK):
    if os.access(home_dir, os.X_OK | os.W_OK):
        TORCH_CPP_BUILD_DIR = TORCH_CPP_BUILD_DIR_BACKUP
    else:
        raise PermissionError('ORTModule could not find a writable directory to cache its internal files.',
                              f'Make {python_package_dir} or {home_dir} writable and try again.')

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
except:
    raise(f'PyTorch {MINIMUM_TORCH_VERSION_STR} must be installed in order to run ONNXRuntime ORTModule frontend!')

# ORTModule must be loaded only after all validation passes
from .ortmodule import ORTModule
