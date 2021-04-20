# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import math
import random
import copy
import torch
from transformers import AutoConfig, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import pytest
from time import sleep
import warnings
from unittest.mock import patch
from collections import OrderedDict
from collections import namedtuple
from inspect import signature

from onnxruntime.training import _utils, ORTModule
import _test_helpers

from torch.nn.parameter import Parameter

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
import numpy as np

def _ortvalue_from_dlpack(dlpack_tensor):
    return OrtValue(C.OrtValue.from_dlpack(dlpack_tensor, False))

def run_with_pytorch_on_gpu(model, input_list, output_shape):
    print('Use PyTorch for CUDA run....')
    device = torch.device('cuda:0')
    model.to(device)
    inputs_on_cuda = [input_.to(device) for input_ in input_list]
    output = model(*inputs_on_cuda)
    criterion = torch.nn.MSELoss()

    target=torch.ones(*output_shape).to(device)
    loss = criterion(output, target)
    loss.backward()
    torch.cuda.synchronize()
    return output, [input_.grad for input_ in inputs_on_cuda if input_.requires_grad is True]

def run_with_ort_on_gpu(model, input_list, output_shape):
    print('Use ORTModule for CUDA run....')
    device = torch.device('cuda:0')
    model.to(device)
    model = ORTModule(copy.deepcopy(model))
    inputs_on_cuda = [input_.to(device) for input_ in input_list]
    output = model(*inputs_on_cuda)
    criterion = torch.nn.MSELoss()

    target=torch.ones(*output_shape).to(device)
    loss = criterion(output, target)
    loss.backward()
    torch.cuda.synchronize()
    return output, [input_.grad for input_ in inputs_on_cuda if input_.requires_grad is True]

@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff*g

class GeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        print("GeLUFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp


class GeLUModel(torch.nn.Module):
    def __init__(self, output_size):
        super(GeLUModel, self).__init__()
        self.relu = GeLUFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()

    def forward(self, model_input):
        out = self.relu(model_input, self.bias)
        return out

def compare_numpy_list(val_a, val_b):
    for np_a, np_b in zip(val_a, val_b):
        equal_ = np.allclose(np_a, np_b, 1e-7, 1e-6, equal_nan=True)
        if equal_ is False:
            print("== details ==")
            k=np_a.reshape(-1)[:100]
            l=np_b.reshape(-1)[:100]
            is_equal = np.isclose(k, l, 1e-7, 1e-6, equal_nan=True)
            res = (is_equal + 1) % 2
            diff_indices = np.nonzero(res)
            print(diff_indices)
            print(k, l)
            raise ValueError("find a diff")

    print("outputs matched successfully.")

def test_GeLU():
    output_size = 1024
    m = GeLUModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    class GeLUFunctionWrapperModule(torch.nn.Module):
        def __init__(self):
            super(GeLUFunctionWrapperModule, self).__init__()
            self.input_tensors = []
            self.forward_outputs = []
            self.output_tensor = None

        def compute(self, x, y):
            try:
                import builtins
                self.input_tensors = [from_dlpack(i) for i in [x, y]]
                # what if custom function modify x, and in ORT is using an unexpected value at the same time.
                for _, t in enumerate(self.input_tensors):
                    t.requires_grad = True
                with torch.enable_grad():
                    print("==== Entering GeLUFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    self.output_tensor = GeLUFunction.apply(*self.input_tensors)
                    # print(self.output_tensor)
                    # print("===== GeLUFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.output_tensor, self.output_tensor.device, self.output_tensor.grad_fn))
                    forward_outputs = [self.output_tensor] #[ret.contiguous()]
                    [print("===== GeLUFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                    # need hold the forward outputs before PythonOp Compute completed.
                    self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                    [print("===== GeLUFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                    ctx_ptr = int(id(self.output_tensor.grad_fn))
                    # ctx_ptr = int(id(ret))
                    #print(self.y.grad_fn.saved_tensors)
                    return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                    print(return_vals)
                    print("==== Exiting GeLUFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    return tuple(return_vals)
            except Exception as e:
                print("GeLUFunctionWrapperModule:", e)
                return []

        def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = GeLUFunction.backward(ctx, self.x_t) # return two outputs
                forward_outputs = list(ret) #[ret.contiguous()] 

                [print("GeLUFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("GeLUFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("GeLUFunctionWrapperModule backward_compute:", e)
                return []

    ort.register_custom_torch_function_forward("GeLUFunction", GeLUFunctionWrapperModule)
    ort.register_custom_torch_function_backward("GeLUFunction", GeLUFunctionWrapperModule)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

###################################################################################

class MegatronFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        print("MegatronFFunction(torch.autograd.Function) forward")
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # Bypass the reduce if we are using only 1 GPU.
        return grad_output

class MegatronFModel(torch.nn.Module):
    def __init__(self, output_size):
        super(MegatronFModel, self).__init__()
        self.copy_ = MegatronFFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()

    def forward(self, model_input):
        model_input = model_input + self.bias
        out = self.copy_(model_input)
        return out

def test_MegatronF():
    output_size = 1024
    m = MegatronFModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    class MegatronFFunctionWrapperModule(torch.nn.Module):
        def __init__(self):
            super(MegatronFFunctionWrapperModule, self).__init__()
            self.input_tensors = []
            self.forward_outputs = []
            self.output_tensor = None

        def compute(self, x):
            try:
                import builtins
                self.input_tensors = [from_dlpack(i) for i in [x]]
                # what if custom function modify x, and in ORT is using an unexpected value at the same time.
                # todo: assign requires_grad according to original attributes.
                for _, t in enumerate(self.input_tensors):
                    t.requires_grad = True
                with torch.enable_grad():
                    print("==== Entering MegatronFFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    self.output_tensor = MegatronFFunction.apply(*self.input_tensors)
                    # print(self.output_tensor)
                    # print("===== MegatronFFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.output_tensor, self.output_tensor.device, self.output_tensor.grad_fn))
                    forward_outputs = [self.output_tensor] #[ret.contiguous()]
                    [print("===== MegatronFFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                    # need hold the forward outputs before PythonOp Compute completed.
                    self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                    [print("===== MegatronFFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                    ctx_ptr = int(id(self.output_tensor.grad_fn))
                    # ctx_ptr = int(id(ret))
                    #print(self.y.grad_fn.saved_tensors)
                    return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                    print(return_vals)
                    print("==== Exiting MegatronFFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    return tuple(return_vals)
            except Exception as e:
                print("MegatronFFunctionWrapperModule:", e)
                return []

        def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = MegatronFFunction.backward(ctx, self.x_t)
                forward_outputs = [ret] #[ret.contiguous()] 

                [print("MegatronFFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("MegatronFFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("MegatronFFunctionWrapperModule backward_compute:", e)
                return []

    ort.register_custom_torch_function_forward("MegatronFFunction", MegatronFFunctionWrapperModule)
    ort.register_custom_torch_function_backward("MegatronFFunction", MegatronFFunctionWrapperModule)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

#############################################################

class SimpleInplaceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, bias):
        print("SimpleInplaceFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(input_, bias)
        # do something inplace for example in cpp extension.
        return input_.add_(3 * bias)

    @staticmethod
    def backward(ctx, grad_output):
        # Bypass the reduce if we are using only 1 GPU.
        return grad_output, grad_output

class SimpleInplaceModel(torch.nn.Module):
    def __init__(self, output_size):
        super(SimpleInplaceModel, self).__init__()
        self.inplace_op_ = SimpleInplaceFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()

    def forward(self, model_input):
        x = model_input.mul(2)
        y1 = self.inplace_op_(x, self.bias)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

def test_SimpleInplace():
    output_size = 1024
    m = SimpleInplaceModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    class SimpleInplaceFunctionWrapperModule(torch.nn.Module):
        def __init__(self):
            super(SimpleInplaceFunctionWrapperModule, self).__init__()
            self.input_tensors = []
            self.forward_outputs = []
            self.output_tensor = None

        def compute(self, x, y):
            try:
                import builtins
                self.input_tensors = [from_dlpack(i) for i in [x, y]]
                # what if custom function modify x, and in ORT is using an unexpected value at the same time.
                # todo: assign requires_grad according to original attributes.
                for _, t in enumerate(self.input_tensors):
                    t.requires_grad = True
                with torch.enable_grad():
                    print("==== Entering SimpleInplaceFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    self.output_tensor = SimpleInplaceFunction.apply(*self.input_tensors)
                    # print(self.output_tensor)
                    # print("===== MegatronFFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.output_tensor, self.output_tensor.device, self.output_tensor.grad_fn))
                    forward_outputs = [self.output_tensor] #[ret.contiguous()]
                    [print("===== SimpleInplaceFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                    # need hold the forward outputs before PythonOp Compute completed.
                    self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                    [print("===== SimpleInplaceFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                    ctx_ptr = int(id(self.output_tensor.grad_fn))
                    # ctx_ptr = int(id(ret))
                    #print(self.y.grad_fn.saved_tensors)
                    return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                    print(return_vals)
                    print("==== Exiting SimpleInplaceFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    return tuple(return_vals)
            except Exception as e:
                print("SimpleInplaceFunctionWrapperModule:", e)
                return []

        def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = SimpleInplaceFunction.backward(ctx, self.x_t)
                forward_outputs = list(ret) #[ret.contiguous()] 

                [print("SimpleInplaceFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("SimpleInplaceFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("SimpleInplaceFunctionWrapperModule backward_compute:", e)
                return []

    ort.register_custom_torch_function_forward("SimpleInplaceFunction", SimpleInplaceFunctionWrapperModule)
    ort.register_custom_torch_function_backward("SimpleInplaceFunction", SimpleInplaceFunctionWrapperModule)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

test_GeLU()
test_MegatronF()
test_SimpleInplace()