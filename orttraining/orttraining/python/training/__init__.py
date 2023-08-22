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
from onnxruntime.capi.training.training_session import TrainingSession


# Options need to be imported before `ORTTrainer`.
from .orttrainer_options import ORTTrainerOptions
from .orttrainer import ORTTrainer, TrainStepInfo
from . import amp, artifacts, checkpoint, model_desc_validation, optim

__all__ = [
    "PropagateCastOpsStrategy",
    "TrainingParameters",
    "is_ortmodule_available",
    "TrainingSession",
    "ORTTrainerOptions",
    "ORTTrainer",
    "TrainStepInfo",
    "amp",
    "artifacts",
    "checkpoint",
    "model_desc_validation",
    "optim",
]

try:
    if is_ortmodule_available():
        from .ortmodule import ORTModule  # noqa: F401

        __all__.append("ORTModule")
except ImportError:
    # That is OK iff this is not a ORTModule training package
    pass
