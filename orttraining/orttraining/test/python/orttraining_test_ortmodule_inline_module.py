import torch
from onnxruntime.training.ortmodule import ORTModule
import onnxruntime

onnxruntime.training.ortmodule.ONNX_OPSET_VERSION = 15
torch.manual_seed(42)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)


class ORTModuleExtension(ORTModule):
    def __init__(self, module, debug_options=None):
        super().__init__(module, debug_options)
        for training_mode in [False, True]:
            self._torch_module._execution_manager(training_mode)._export_extra_kwargs = {
                "export_modules_as_functions": {M}
            }


model = ORTModuleExtension(M())
model(torch.randn(1, 1))
