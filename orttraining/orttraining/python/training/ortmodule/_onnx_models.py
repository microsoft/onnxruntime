# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _onnx_models.py

from dataclasses import dataclass
from filelock import SoftFileLock
import onnx
import os
import torch
import warnings

def _get_onnx_file_name(prefix, name, export_mode):
    suffix = 'training' if export_mode == torch.onnx.TrainingMode.TRAINING else 'inference'
    return f"{prefix}_{name}_{suffix}.onnx"

def _save_model(model: onnx.ModelProto, file_path: str):
    print("saving file at ", file_path)
    with SoftFileLock(file_path+'.lock'):
        # SoftFileLock ensures that multiple processes do not compete with each other
        # for writing priveleges by using locks by checking the existence of the lock file.
        try:
            onnx.save(model, file_path)
        except Exception as exc:
            # If the file saving fails, it is possible that the lock file will be left
            # behind in the directory provided by the user.
            # Warn the user to delete that file before proceeding.
            warnings.warn(f"There was an error while saving the onnx model. Please clear the file {file_path}.lock before retrying.")
            raise exc

@dataclass
class OnnxModels:
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
