import onnx
import torch
torch.manual_seed(1)
from onnxruntime.training import ORTModule
import onnxruntime as ort
import os
from torch.utils.dlpack import from_dlpack, to_dlpack
 
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C
import copy

def _ortvalue_from_dlpack(dlpack_tensor):
    return OrtValue(C.OrtValue.from_dlpack(dlpack_tensor, False))

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
 
    @staticmethod
    def backward(ctx, grad_output):
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
 
class CustomFnWrapperModule(torch.nn.Module):
    def __init__(self):
        super(CustomFnWrapperModule, self).__init__()
        self.x_t = None
        self.forward_outputs = []
        self.y = None

    def compute(self, x):
        try:
            self.x_t = from_dlpack(x)
            # what if custom function modify x, and in ORT is using an unexpected value at the same time.
            self.x_t.requires_grad = True
            with torch.enable_grad():
                print("==== Entering CustomFnWrapperModule.compute , process id {} ====".format(os.getpid()))
                self.y = MyReLU.apply(self.x_t)
                print(self.y)
                print("===== CustomFnWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.y, self.y.device, self.y.grad_fn))
                forward_outputs = [self.y] #[ret.contiguous()]
                [print("===== CustomFnWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                # need hold the forward outputs before PythonOp Compute completed.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                [print("===== CustomFnWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                ctx_ptr = int(id(self.y.grad_fn))
                # ctx_ptr = int(id(ret))
                #print(self.y.grad_fn.saved_tensors)
                return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                print("==== Exiting CustomFnWrapperModule.compute , process id {} ====".format(os.getpid()))
                return tuple(return_vals)
        except Exception as e:
            print(e)
            return []

    def forward(self, x):
        return MyReLU.apply(x)

    def backward_compute(self, ctx, x):
        print(ctx, ctx.saved_tensors)
        self.x_t = from_dlpack(x)
        self.x_t.requires_grad = False
        
        ret = MyReLU.backward(ctx, self.x_t)
        forward_outputs = [ret] #[ret.contiguous()]
        [print("CustomFnWrapperModule.backward_compute: shape: ", a.shape) for a in forward_outputs]
        # need hold the forward outputs before PythonOp Compute completed.
        self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
        [print("CustomFnWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

        return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
        print(return_vals)
        return tuple(return_vals)


ort.register_custom_torch_function_forward("MyReLU", CustomFnWrapperModule)
ort.register_custom_torch_function_backward("MyReLU", CustomFnWrapperModule)
 
# Define input.
x = torch.tensor([[0.3971, 0.7544],
                  [0.5695, 0.4388]], requires_grad=True)

model = MyModule()
torch.onnx.export(model, (x,), 'model.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, custom_opsets={"prim":1})
onnx_model = onnx.load('model.onnx')
print(onnx_model)
print('-------------------------------------------')

def run_with_pytorch_on_cpu(model):
    print('Use PyTorch for CPU run....')
    y = model(x)
    y.sum().backward()
    print('x:\n', x)
    print('x.grad:\n', x.grad)
    print('y:\n', y)
    return x, x.grad, y

def run_with_ort_on_cpu(model):
    print('Use ORTModule for CPU run....')
    model = ORTModule(copy.deepcopy(model))
    y = model(x)
    y.sum().backward()
    print('x:\n', x)
    print('x.grad:\n', x.grad)
    print('y:\n', y)
    return x, x.grad, y


def run_with_pytorch_on_gpu(model):
    print('Use PyTorch for CUDA run....')
    device = torch.device('cuda:0')
    model.to(device)
    model = ORTModule(copy.deepcopy(model))
    y = model(x.to(device))
    y.sum().backward()
    print('x:\n', x)
    print('x.grad:\n', x.grad)
    print('y:\n', y)
    torch.cuda.synchronize()
    return x, x.grad, y

def run_with_ort_on_gpu(model):
    print('Use ORTModule for CUDA run....')
    device = torch.device('cuda:0')
    model.to(device)
    model = ORTModule(copy.deepcopy(model))
    y = model(x.to(device))
    y.sum().backward()
    print('x:\n', x)
    print('x.grad:\n', x.grad)
    print('y:\n', y)
    torch.cuda.synchronize()
    return x, x.grad, y

_, x_grad1, y1 = run_with_pytorch_on_cpu(model)
_, x_grad2, y2 = run_with_ort_on_cpu(model)
cpu_x_grad_equal = torch.all(torch.eq(x_grad1, x_grad2))
cpu_y_equal = torch.all(torch.eq(y1, y2))

_, x_grad1, y1 = run_with_pytorch_on_gpu(model)
_, x_grad2, y2 = run_with_ort_on_gpu(model)
gpu_x_grad_equal = torch.all(torch.eq(x_grad1, x_grad2))
gpu_y_equal = torch.all(torch.eq(y1, y2))

print("cpu_x_grad_equal: {}, cpu_y_equal: {}, gpu_x_grad_equal: {}, gpu_y_equal: {}".format(bool(cpu_x_grad_equal), bool(cpu_y_equal), bool(gpu_x_grad_equal), bool(gpu_y_equal)))