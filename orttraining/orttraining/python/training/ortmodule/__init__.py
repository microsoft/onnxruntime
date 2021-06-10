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

# Verify proper PyTorch is installed before proceding to ONNX Runtime initialization
try:
    import torch
    torch_version = version.parse(torch.__version__.split('+')[0])
    minimum_torch_version = version.parse(MINIMUM_TORCH_VERSION_STR)
    if torch_version < minimum_torch_version:
        raise RuntimeError(
            f'ONNX Runtime ORTModule frontend requires PyTorch version greater or equal to {MINIMUM_TORCH_VERSION_STR}, '
            f'but version {torch.__version__} was found instead.')
except:
    raise(f'PyTorch {MINIMUM_TORCH_VERSION_STR} must be installed in order to run ONNX Runtime ORTModule frontend!')

from ._custom_autograd_function import enable_custom_autograd_support
enable_custom_autograd_support()

# Initalized ORT's random seed with pytorch's initial seed
# Initalized ORT's random seed with pytorch's current seed, 
# in case user has set pytorch seed before importing ORTModule
import sys
from onnxruntime import set_seed
set_seed((torch.initial_seed() % sys.maxsize))

# Override torch.manual_seed and torch.cuda.manual_seed
def override_torch_manual_seed(seed):
    set_seed(seed % sys.maxsize)
    return torch_manual_seed(seed)
torch_manual_seed = torch.manual_seed
torch.manual_seed = override_torch_manual_seed

def override_torch_cuda_manual_seed(seed):
    set_seed(seed % sys.maxsize)
    return torch_cuda_manual_seed(seed)
torch_cuda_manual_seed = torch.cuda.manual_seed
torch.cuda.manual_seed = override_torch_cuda_manual_seed

# ORTModule must be loaded only after all validation passes
from .ortmodule import ORTModule
