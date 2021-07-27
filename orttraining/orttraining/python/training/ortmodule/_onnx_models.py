# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _onnx_models.py

from dataclasses import dataclass
from filelock import SoftFileLock
import onnx
import os
import torch

def _get_onnx_file_name(prefix, name, export_mode):
    suffix = 'training' if export_mode == torch.onnx.TrainingMode.TRAINING else 'inference'
    return f"{prefix}_{name}_{suffix}.onnx"

def _save_model(model: onnx.ModelProto, file_path: str):
    onnx.save(model, file_path)

@dataclass
class ONNXModels:
    """Encapsulates all ORTModule onnx models."""

    exported_model: onnx.ModelProto = None
    optimized_model: onnx.ModelProto = None

    def save_exported_model(self, directory, prefix, export_mode):
        # save the ortmodule exported model
        _save_model(self.exported_model, os.path.join(directory,
                                                      _get_onnx_file_name(prefix, 'torch_exported', export_mode)))

    def save_optimized_model(self, directory, prefix, export_mode):
        # save the ortmodule optimized model
        _save_model(self.optimized_model, os.path.join(directory,
                                                       _get_onnx_file_name(prefix, 'optimized', export_mode)))
