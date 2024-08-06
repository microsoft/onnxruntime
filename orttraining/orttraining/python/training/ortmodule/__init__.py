# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import contextlib
import inspect
import os
import sys
import warnings

import torch
from packaging import version

from onnxruntime import set_seed
from onnxruntime.capi import build_and_package_info as ort_info
from onnxruntime.capi._pybind_state import is_ortmodule_available

from ._fallback import ORTModuleFallbackException, ORTModuleInitException, _FallbackPolicy, wrap_exception
from .torch_cpp_extensions import is_installed as is_torch_cpp_extensions_installed

if not is_ortmodule_available():
    raise ImportError("ORTModule is not supported on this platform.")


def _defined_from_envvar(name: str, default_value: any, warn: bool = True):
    """Check given name exists in the environment variable and return the value using the default_value's
    type if it exists.
    """
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


def _override_gradient_checkpoint(original_checkpoint):
    """
    Best effort to override `torch.utils.checkpoint` and `deepspeed.checkpointing.checkpoint` during ONNX export.

    Despite importing `torch.utils.checkpoint` or `deepspeed.checkpointing.checkpoint` in `__init__.py`,
    users might import it first, causing our override to not take effect. We still attempt to override
    it to work in most cases.

    We replace the checkpoint function with our implementation, without condition checks.
    The actual check is in the overridden function, verifying if:
    1) `checkpoint` is called during ORTModule model export,
    2) Gradient checkpoint autograd function is disallowed (ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT),
    3) Memory optimization level is not specified by the user (ORTMODULE_MEMORY_OPT_LEVEL).
    If true, we reset memory optimization to layer-wise recompute.

    """

    # Note: The `torch.utils.checkpoint` checkpoint function signature looks like below:
    #   `checkpoint(function, *args,
    #               use_reentrant = None,
    #               context_fn = noop_context_fn,
    #               determinism_check = _DEFAULT_DETERMINISM_MODE,
    #               debug = False,
    #               **kwargs).`
    # The few keyword arguments are not used in the recompute module forward function, but by the
    # checkpoint function itself, so we need to filter them out otherwise module forward function
    # would complain about unexpected keyword arguments.
    all_input_parameters = inspect.signature(original_checkpoint).parameters.values()
    outside_kwarg_params = []
    for input_parameter in all_input_parameters:
        if (
            input_parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            or input_parameter.kind == inspect.Parameter.KEYWORD_ONLY
            or input_parameter.kind == inspect.Parameter.VAR_KEYWORD
        ):
            outside_kwarg_params.append(input_parameter.name)

    def _checkpoint(
        function,
        *args,
        **kwargs,
    ):
        # Conditions to activate layer-wise memory optimization automatically:
        # 1. Checkpoint is called during ORTModule model export context.
        # 2. Gradient checkpoint autograd function export is disallowed.
        # 3. Memory optimization level is layer-wise recompute.
        if (
            ORTMODULE_ONNX_EXPORT_CONTEXT[0] is True
            and _defined_from_envvar("ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT", 0) != 1
            and _defined_from_envvar("ORTMODULE_MEMORY_OPT_LEVEL", 0) == 1
        ):
            for name in outside_kwarg_params:
                if name in kwargs:
                    # Pop out the keyword argument to avoid passing it to the module run function
                    kwargs.pop(name)
            print(
                "Layer-wise memory optimization is enabled upon detecting "
                "gradient checkpointing autograd function usage during model execution."
            )
            return function(*args, **kwargs)
        return original_checkpoint(
            function,
            *args,
            **kwargs,
        )

    return _checkpoint


with contextlib.suppress(Exception):
    from torch.utils.checkpoint import checkpoint as original_torch_checkpoint

    torch.utils.checkpoint.checkpoint = _override_gradient_checkpoint(original_torch_checkpoint)

    import deepspeed

    original_deepspeed_checkpoint = deepspeed.checkpointing.checkpoint
    deepspeed.checkpointing.checkpoint = _override_gradient_checkpoint(original_deepspeed_checkpoint)


################################################################################
# All global constant goes here, before ORTModule is imported ##################
# NOTE: To *change* values in runtime, import onnxruntime.training.ortmodule and
# assign them new values. Importing them directly do not propagate changes.
################################################################################
ONNX_OPSET_VERSION = 17
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

# The first value indicates whether the code is in ONNX export context.
# The export context here include the full export process, including prepare export input/output information,
# and export model.
ORTMODULE_ONNX_EXPORT_CONTEXT = [False]


@contextlib.contextmanager
def export_context():
    """Context manager for model export."""
    try:
        ORTMODULE_ONNX_EXPORT_CONTEXT[0] = True

        yield
    finally:
        ORTMODULE_ONNX_EXPORT_CONTEXT[0] = False


# Verify minimum PyTorch version is installed before proceeding to ONNX Runtime initialization
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

# Initialized ORT's random seed with pytorch's initial seed
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


from .graph_optimizer_registry import register_graph_optimizer  # noqa: E402, F401
from .graph_optimizers import *  # noqa: E402, F403
from .options import DebugOptions, LogLevel  # noqa: E402, F401

# ORTModule must be loaded only after all validation passes
from .ortmodule import ORTModule  # noqa: E402, F401
