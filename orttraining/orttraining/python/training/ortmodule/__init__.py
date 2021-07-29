# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys

from glob import glob
from packaging import version

from ._fallback import _FallbackManager, ORTModuleInitException, wrap_exception


################################################################################
# All global constant goes here, before ORTModule is imported ##################
################################################################################
ONNX_OPSET_VERSION = 12
MINIMUM_RUNTIME_PYTORCH_VERSION_STR = '1.8.1'
TORCH_CPP_DIR = os.path.join(os.path.dirname(__file__),
                             'torch_cpp_extensions')

# Verify minimum PyTorch version is installed before proceding to ONNX Runtime initialization
try:
    import torch
    runtime_pytorch_version = version.parse(torch.__version__.split('+')[0])
    minimum_runtime_pytorch_version = version.parse(MINIMUM_RUNTIME_PYTORCH_VERSION_STR)
    if runtime_pytorch_version < minimum_runtime_pytorch_version:
        raise wrap_exception(ORTModuleInitException,
                             RuntimeError(
                                 f'ONNX Runtime ORTModule frontend requires PyTorch version greater or equal to {MINIMUM_RUNTIME_PYTORCH_VERSION_STR}, '
                                 f'but version {torch.__version__} was found instead.'))
except:
    raise wrap_exception(ORTModuleInitException,
                         RuntimeError(f'PyTorch {MINIMUM_RUNTIME_PYTORCH_VERSION_STR} must be installed in order to run ONNX Runtime ORTModule frontend!'))

# Verify whether PyTorch C++ extensions are already compiled
torch_cpp_exts = glob(os.path.join(TORCH_CPP_DIR, '*.so'))
torch_cpp_exts.extend(glob(os.path.join(TORCH_CPP_DIR, '*.dll')))
torch_cpp_exts.extend(glob(os.path.join(TORCH_CPP_DIR, '*.dylib')))
if not torch_cpp_exts and '-m' not in sys.argv:
    raise wrap_exception(ORTModuleInitException,
                         EnvironmentError(f"ORTModule's extensions were not detected at '{TORCH_CPP_DIR}' folder. "
                                         "Run `python -m torch_ort.configure` before using `ORTModule` frontend."))

# PyTorch custom Autograd function support
from ._custom_autograd_function import enable_custom_autograd_support
enable_custom_autograd_support()

# Initalized ORT's random seed with pytorch's initial seed
# in case user has set pytorch seed before importing ORTModule
import sys
from onnxruntime import set_seed
set_seed((torch.initial_seed() % sys.maxsize))

# Override torch.manual_seed and torch.cuda.manual_seed
def override_torch_manual_seed(seed):
    set_seed(int(seed % sys.maxsize))
    return torch_manual_seed(seed)
torch_manual_seed = torch.manual_seed
torch.manual_seed = override_torch_manual_seed

def override_torch_cuda_manual_seed(seed):
    set_seed(int(seed % sys.maxsize))
    return torch_cuda_manual_seed(seed)
torch_cuda_manual_seed = torch.cuda.manual_seed
torch.cuda.manual_seed = override_torch_cuda_manual_seed

# ORTModule must be loaded only after all validation passes
from .ortmodule import ORTModule
from .debug_options import DebugOptions, LogLevel
