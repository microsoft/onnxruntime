import os
from torch.utils.dlpack import from_dlpack, to_dlpack
import onnxruntime
import numpy as np
import torch
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C

def _ortvalue_from_dlpack(dlpack_tensor):
    return OrtValue(C.OrtValue.from_dlpack(dlpack_tensor, False))

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        print("saved a tensor in ctx in forward pass, the tensor is ", input)
        output = input * 2 #input.clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        print("get a tensor in ctx in backward pass, the tensor is ", input)
        return grad_input


class CustomFnWrapperModule(torch.nn.Module):
    def __init__(self, A,B,C):
        super(CustomFnWrapperModule, self).__init__()
        self.a,self.b,self.c = A,B,C
        self.x_t = None
        self.contiguous_grad_outputs = []

    def compute(self, x):
        try:
            self.x_t = from_dlpack(x)
            # what if custom function modify x, and in ORT is using an unexpected value at the same time.
            self.x_t.requires_grad = True
            print("Current process id is ", os.getpid())
            (ret) = self.forward(self.x_t)
            print("device: ", ret.device)
            v = ret.data_ptr()
            print("address: ", v)
            grad_output = ret.contiguous()
            self.contiguous_grad_outputs.append(grad_output)
            self.contiguous_grad_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in self.contiguous_grad_outputs]
            return_vals = [r.ortvalue_ptr() for r in self.contiguous_grad_outputs]
            # 235 is the fake address of ctx.grad_func.
            return_vals = [235] + return_vals
            print(return_vals)
            return tuple(return_vals)
        except Exception as e:
            print(e)
            return []

    def forward(self, x):
        return MyReLU.apply(x)