# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys

from packaging import version

from ._fallback import (_FallbackManager,
                        _FallbackPolicy,
                        ORTModuleFallbackException,
                        ORTModuleInitException,
                        wrap_exception)
from .torch_cpp_extensions import is_installed as is_torch_cpp_extensions_installed


################################################################################
# All global constant goes here, before ORTModule is imported ##################
################################################################################
ONNX_OPSET_VERSION = 12
MINIMUM_RUNTIME_PYTORCH_VERSION_STR = '1.8.1'
TORCH_CPP_DIR = os.path.join(os.path.dirname(__file__),
                             'torch_cpp_extensions')
_FALLBACK_INIT_EXCEPTION = None
ORTMODULE_FALLBACK_POLICY = _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE |\
                            _FallbackPolicy.FALLBACK_UNSUPPORTED_DATA |\
                            _FallbackPolicy.FALLBACK_UNSUPPORTED_TORCH_MODEL |\
                            _FallbackPolicy.FALLBACK_UNSUPPORTED_ONNX_MODEL |\
                            _FallbackPolicy.FALLBACK_BAD_INITIALIZATION
ORTMODULE_FALLBACK_RETRY = False

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
except ORTModuleFallbackException as e:
    # Initialization fallback is handled at ORTModule.__init__
    _FALLBACK_INIT_EXCEPTION = e
except ImportError as e:
    raise RuntimeError(f'PyTorch {MINIMUM_RUNTIME_PYTORCH_VERSION_STR} must be installed in order to run ONNX Runtime ORTModule frontend!') from e

# Verify whether PyTorch C++ extensions are already compiled

if not is_torch_cpp_extensions_installed(TORCH_CPP_DIR) and '-m' not in sys.argv:
    _FALLBACK_INIT_EXCEPTION = wrap_exception(ORTModuleInitException,
                                                        EnvironmentError(
                                                            f"ORTModule's extensions were not detected at '{TORCH_CPP_DIR}' folder. "
                                                            "Run `python -m torch_ort.configure` before using `ORTModule` frontend."))

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
