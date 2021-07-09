# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.capi._pybind_state import TrainingParameters
from onnxruntime.capi._pybind_state import PropagateCastOpsStrategy
from onnxruntime.capi.training.training_session import TrainingSession

from .orttrainer_options import ORTTrainerOptions
from .orttrainer import ORTTrainer, TrainStepInfo
from . import amp, checkpoint, optim, model_desc_validation


try:
    from .ortmodule import ORTModule
except ImportError:
    # Not a ORTModule training package
    pass
except Exception as e:
    try:
        from onnxruntime.training.ortmodule._fallback import ORTModuleInitException
        if isinstance(e, ORTModuleInitException):
            # ORTModule is present but not ready to run
            # That is OK when this is not a ORTModule training package
            pass
    except Exception:
        # Not a ORTModule training package
        pass
