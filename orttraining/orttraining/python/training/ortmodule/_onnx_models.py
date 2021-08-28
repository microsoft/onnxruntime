# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _onnx_models.py

from dataclasses import dataclass
import onnx
import os
import torch

def _get_onnx_file_name(name_prefix, name, export_mode):
    suffix = 'training' if export_mode == torch.onnx.TrainingMode.TRAINING else 'inference'
    return f"{name_prefix}_{name}_{suffix}.onnx"

def _save_model(model: onnx.ModelProto, file_path: str):
    onnx.save(model, file_path)

@dataclass
class ONNXModels:
    """Encapsulates all ORTModule onnx models.

    1. exported_model: Model that is exported by torch.onnx.export
    2. optimized_model: Optimized model after gradients graph has been built.
    3. optimized_pre_grad_model: Optimized model before gradient graph is built.
    4. In addition, ORTModule also saves the execution_model which is the model
       that is being executed by ORT. It has further optimizations done by the
       InferenceSession and is saved by the InferenceSession.
    """

    exported_model: onnx.ModelProto = None
    optimized_model: onnx.ModelProto = None
    optimized_pre_grad_model: onnx.ModelProto = None

    def save_exported_model(self, path, name_prefix, export_mode):
        # save the ortmodule exported model
        _save_model(self.exported_model,
                    os.path.join(path, _get_onnx_file_name(name_prefix, 'torch_exported', export_mode)))

    def save_optimized_model(self, path, name_prefix, export_mode):
        # save the ortmodule optimized model
        _save_model(self.optimized_model,
                    os.path.join(path, _get_onnx_file_name(name_prefix, 'optimized', export_mode)))
        _save_model(self.optimized_pre_grad_model,
                    os.path.join(path, _get_onnx_file_name(name_prefix, 'optimized_pre_grad', export_mode)))
