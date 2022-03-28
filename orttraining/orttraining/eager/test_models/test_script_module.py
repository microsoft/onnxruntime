import torch
import onnxruntime
from onnxruntime.training import ORTModule
from onnxruntime.training.ortmodule import DebugOptions
from onnxruntime.training.ortmodule import _io

class MMNet(torch.nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()
        self.layer=torch.nn.LayerNorm(256)

    def forward(self, input):
        return self.layer(input)
input1 = torch.ones(2, 256, dtype=torch.float32)
mm = MMNet()
device = torch.device('ort')
#device = torch.device('cpu')
mm = mm.to(device)
mm = torch.jit.script(mm)
mm.__dict__['_original_module'] = mm
mm.output_names = ["output-0"]
mm.output_dynamic_axes={'output-0': {0: 'dim1', 1: 'dim2'}}
mm.module_output_schema=_io._TensorStub(dtype=torch.float32, shape_dims=2)

mm = ORTModule(mm, DebugOptions(save_onnx=True, onnx_prefix='my_model'))
input1 = input1.to(device)
input1.contiguous()
output = mm(input1)
print(output.cpu())

