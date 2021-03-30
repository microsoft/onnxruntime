import onnx
import torch
torch.manual_seed(1)
from onnxruntime.training import ORTModule
import onnxruntime as ort
import os
from torch.utils.dlpack import from_dlpack, to_dlpack
 
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C

def _ortvalue_from_dlpack(dlpack_tensor):
    return OrtValue(C.OrtValue.from_dlpack(dlpack_tensor, False))

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        print('MyReLU forward.')
        # what if custom function modify x, and in ORT is using an unexpected value at the same time.
        print("Current process id is ", os.getpid())
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


class CustomFnWrapperModule(torch.nn.Module):
    def __init__(self):
        super(CustomFnWrapperModule, self).__init__()
        self.x_t = None
        self.forward_outputs = []

    def compute(self, x):
        try:
            self.x_t = from_dlpack(x)
            # what if custom function modify x, and in ORT is using an unexpected value at the same time.
            self.x_t.requires_grad = True
            print("Current process id is ", os.getpid())
            (ret) = self.forward(self.x_t)
            print("device: ", ret.device)
            v = ret.data_ptr()
            print("v : ", v)
            forward_outputs = [ret] #[ret.contiguous()]
            [print("CustomFnWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]
            # need hold the forward outputs before PythonOp Compute completed.
            self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
            [print("CustomFnWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

            ctx_ptr = int(id(ret.grad_fn))
            return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
            print(return_vals)
            return tuple(return_vals)
        except Exception as e:
            print(e)
            return []

    def forward(self, x):
        return MyReLU.apply(x)


# def forward_wrapper(x):
#     x_t = from_dlpack(x)
#     x_t.requires_grad = True
#     ret = MyReLU.apply(x_t)
#     ctx_ptr = int(id(ret.grad_fn))
#     packed_val = _ortvalue_from_dlpack(torch.utils.dlpack.to_dlpack(ret))
#     return_vals = [ctx_ptr, int(packed_val.ortvalue_ptr())]
#     print(return_vals)

#     return tuple(return_vals)

ort.register_custom_torch_function_forward("MyReLU", CustomFnWrapperModule)
ort.register_custom_torch_function_backward("MyReLU", MyReLU.backward)
 
y = model(x)
 
y.sum().backward()
 
print('x:\n', x)
print('x.grad:\n', x.grad)
print('y:\n', y)