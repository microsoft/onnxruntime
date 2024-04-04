# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# isort: skip_file
from onnxruntime.capi._pybind_state import (
    PropagateCastOpsStrategy,
    TrainingParameters,
    is_ortmodule_available,
)

# Options need to be imported before `ORTTrainer`.
from . import amp, artifacts, optim

__all__ = [
    "PropagateCastOpsStrategy",
    "TrainingParameters",
    "is_ortmodule_available",
    "amp",
    "artifacts",
    "optim",
]

try:
    if is_ortmodule_available():
        from .ortmodule import ORTModule

        __all__ += ["ORTModule"]
except ImportError:
    # That is OK iff this is not a ORTModule training package
    pass
