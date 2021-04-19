# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


# Verify proper PyTorch is installed before proceding to ONNX Runtime initializetion
from packaging import version
MINIMUM_TORCH_VERSION_STR = '1.8.1'
try:
    import torch
    torch_version = version.parse(torch.__version__.split('+')[0])
    minimum_torch_version = version.parse(MINIMUM_TORCH_VERSION_STR)
    if torch_version < minimum_torch_version:
        raise RuntimeError(
            f'ONNXRuntime ORTModule frontend requires PyTorch version greater or equal to {MINIMUM_TORCH_VERSION_STR}, '
            f'but version {torch.__version__} was found instead!')
except:
    raise(f'PyTorch {MINIMUM_TORCH_VERSION_STR} must be installed in order to run ONNXRuntime ORTModule frontend!')


from onnxruntime.capi._pybind_state import TrainingParameters
from onnxruntime.capi.training.training_session import TrainingSession

from .orttrainer_options import ORTTrainerOptions
from .orttrainer import ORTTrainer, TrainStepInfo
from . import amp, checkpoint, optim, model_desc_validation
from .execution_agent import InferenceAgent, TrainingAgent
from .ortmodule import ORTModule
from .runstateinfo import RunStateInfo
