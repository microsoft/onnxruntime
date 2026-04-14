import os
import tempfile
import unittest.mock

import torch

from onnxruntime.training.ortmodule import DebugOptions, LogLevel, ORTModule

torch.distributed.init_process_group(backend="nccl")


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.nn.functional.relu(self.fc(x))
        return x


data = torch.randn(1, 10)

with tempfile.TemporaryDirectory() as temporary_dir:
    os.environ["ORTMODULE_CACHE_DIR"] = temporary_dir

    # first time seeing the model, architecture should be cached under ORTMODULE_CACHE_DIR
    model_pre_cache = Net()
    model_pre_cache = ORTModule(model_pre_cache, DebugOptions(log_level=LogLevel.INFO))

    torch.onnx.export = unittest.mock.MagicMock(side_effect=torch.onnx.export)
    _ = model_pre_cache(data)
    torch.onnx.export.assert_called()
    torch.onnx.export.reset_mock()

    # second time seeing the model, architecture should be loaded from ORTMODULE_CACHE_DIR
    model_post_cache = Net()
    model_post_cache = ORTModule(model_post_cache, DebugOptions(log_level=LogLevel.INFO))

    torch.onnx.export = unittest.mock.MagicMock(side_effect=torch.onnx.export)
    _ = model_post_cache(data)
    torch.onnx.export.assert_not_called()
    torch.onnx.export.reset_mock()

    del os.environ["ORTMODULE_CACHE_DIR"]
