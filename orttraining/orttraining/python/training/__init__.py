# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# isort: skip_file
import importlib.util

from onnxruntime.capi._pybind_state import (
    PropagateCastOpsStrategy,
    TrainingParameters,
    is_ortmodule_available,
)

# Options need to be imported before `ORTTrainer`.
from . import amp, artifacts

__all__ = [
    "PropagateCastOpsStrategy",
    "TrainingParameters",
    "amp",
    "artifacts",
    "is_ortmodule_available",
]

if importlib.util.find_spec("torch") is not None:
    from . import optim

    __all__ += ["optim"]

try:
    if is_ortmodule_available():
        from .ortmodule import ORTModule

        __all__ += ["ORTModule"]
except ImportError:
    # That is OK iff this is not a ORTModule training package
    pass
