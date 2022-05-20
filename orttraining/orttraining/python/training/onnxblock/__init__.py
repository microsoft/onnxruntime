# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# __init__.py

from .model import Model, TrainingModel
from .checkpoint_utils import save_checkpoint
from . import loss, optim
from .model_accessor import onnx_model

import onnx

_producer_name = "onnxblock offline tooling"
_opset_import = onnx.helper.make_opsetid("com.microsoft", 1)
