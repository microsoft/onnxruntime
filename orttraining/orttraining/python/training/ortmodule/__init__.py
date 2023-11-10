# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import warnings

import torch
from packaging import version

from onnxruntime import set_seed
from onnxruntime.capi import build_and_package_info as ort_info

from ._fallback import ORTModuleFallbackException, ORTModuleInitException, _FallbackPolicy, wrap_exception
from .torch_cpp_extensions import is_installed as is_torch_cpp_extensions_installed


def _defined_from_envvar(name, default_value, warn=True):
    new_value = os.getenv(name, None)
    if new_value is None:
        return default_value
    try:
        new_value = type(default_value)(new_value)
    except (TypeError, ValueError) as e:
        if warn:
            warnings.warn(f"Unable to overwrite constant {name!r} due to {e!r}.")
        return default_value
    return new_value


################################################################################
# All global constant goes here, before ORTModule is imported ##################
# NOTE: To *change* values in runtime, import onnxruntime.training.ortmodule and
# assign them new values. Importing them directly do not propagate changes.
################################################################################
ONNX_OPSET_VERSION = 15
MINIMUM_RUNTIME_PYTORCH_VERSION_STR = "1.8.1"
ORTMODULE_TORCH_CPP_DIR = os.path.join(os.path.dirname(__file__), "torch_cpp_extensions")
_FALLBACK_INIT_EXCEPTION = None
ORTMODULE_FALLBACK_POLICY = (
    _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE
    | _FallbackPolicy.FALLBACK_UNSUPPORTED_DATA
    | _FallbackPolicy.FALLBACK_UNSUPPORTED_TORCH_MODEL
    | _FallbackPolicy.FALLBACK_UNSUPPORTED_ONNX_MODEL
)
ORTMODULE_FALLBACK_RETRY = False
ORTMODULE_IS_DETERMINISTIC = torch.are_deterministic_algorithms_enabled()

ONNXRUNTIME_CUDA_VERSION = ort_info.cuda_version if hasattr(ort_info, "cuda_version") else None
ONNXRUNTIME_ROCM_VERSION = ort_info.rocm_version if hasattr(ort_info, "rocm_version") else None

# Verify minimum PyTorch version is installed before proceding to ONNX Runtime initialization
try:
    import torch

    runtime_pytorch_version = version.parse(torch.__version__.split("+")[0])
    minimum_runtime_pytorch_version = version.parse(MINIMUM_RUNTIME_PYTORCH_VERSION_STR)
    if runtime_pytorch_version < minimum_runtime_pytorch_version:
        raise wrap_exception(
            ORTModuleInitException,
            RuntimeError(
                "ONNX Runtime ORTModule frontend requires PyTorch version greater"
                f" or equal to {MINIMUM_RUNTIME_PYTORCH_VERSION_STR},"
                f" but version {torch.__version__} was found instead."
            ),
        )
except ORTModuleFallbackException as e:
    # Initialization fallback is handled at ORTModule.__init__
    _FALLBACK_INIT_EXCEPTION = e
except ImportError as e:
    raise RuntimeError(
        f"PyTorch {MINIMUM_RUNTIME_PYTORCH_VERSION_STR} must be "
        "installed in order to run ONNX Runtime ORTModule frontend!"
    ) from e

# Verify whether PyTorch C++ extensions are already compiled
# TODO: detect when installed extensions are outdated and need reinstallation. Hash? Version file?
if not is_torch_cpp_extensions_installed(ORTMODULE_TORCH_CPP_DIR) and "-m" not in sys.argv:
    _FALLBACK_INIT_EXCEPTION = wrap_exception(
        ORTModuleInitException,
        RuntimeError(
            f"ORTModule's extensions were not detected at '{ORTMODULE_TORCH_CPP_DIR}' folder. "
            "Run `python -m torch_ort.configure` before using `ORTModule` frontend."
        ),
    )

# Initalized ORT's random seed with pytorch's initial seed
# in case user has set pytorch seed before importing ORTModule
set_seed(torch.initial_seed() % sys.maxsize)


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


def _use_deterministic_algorithms(enabled):
    global ORTMODULE_IS_DETERMINISTIC  # noqa: PLW0603
    ORTMODULE_IS_DETERMINISTIC = enabled


def _are_deterministic_algorithms_enabled():
    global ORTMODULE_IS_DETERMINISTIC  # noqa: PLW0602
    return ORTMODULE_IS_DETERMINISTIC


from .graph_transformer_registry import register_graph_transformer  # noqa: E402, F401
from .options import DebugOptions, LogLevel  # noqa: E402, F401

# ORTModule must be loaded only after all validation passes
from .ortmodule import ORTModule  # noqa: E402, F401
