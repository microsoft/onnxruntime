import onnx
import torch
torch.manual_seed(1)
from onnxruntime.training import ORTModule
import onnxruntime as ort
 
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        print('MyReLU forward.')
        ctx.save_for_backward(input)
        return input.clamp(min=0)
 
    @staticmethod
    def backward(ctx, grad_output):
        print('MyReLU backward.')
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
 
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_a = torch.nn.Linear(2, 2)
        self.linear_b = torch.nn.Linear(2, 2)
        print('----------------------------')
        print('[dev.py]', type(MyReLU).__name__)
        print('[dev.py]', dir(MyReLU))
        print('----------------------------')
        self.relu = MyReLU.apply# Use MyReLU.apply to fail ORTModule.
 
    def forward(self, x):
        h = self.linear_a(x)
        h = self.relu(h)
        h = self.linear_b(h)
        return h
 
# Define input.
x = torch.tensor([[0.3971, 0.7544],
                  [0.5695, 0.4388]], requires_grad=True)
x.requires_grad = True
model = MyModule()
torch.onnx.export(model, (x,), 'model.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, custom_opsets={"prim":1})
onnx_model = onnx.load('model.onnx')
print(onnx_model)
print('-------------------------------------------')
print('-------------------------------------------')
print('Use ORTModule')
model = ORTModule(model)

ort.register_custom_torch_function_forward("MyReLU", MyReLU.apply)
ort.register_custom_torch_function_backward("MyReLU", MyReLU.backward)
 
y = model(x)
 
y.sum().backward()
 
print('x:\n', x)
print('x.grad:\n', x.grad)
print('y:\n', y)