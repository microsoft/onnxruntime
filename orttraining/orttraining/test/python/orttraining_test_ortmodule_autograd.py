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
    model = copy.deepcopy(model)
    model.to(device)
    model = ORTModule(model)
    inputs_on_cuda = [input_.to(device) for input_ in input_list]
    output = model(*inputs_on_cuda)
    criterion = torch.nn.MSELoss()

    target=torch.ones(*output_shape).to(device)
    loss = criterion(output, target)
    loss.backward()
    torch.cuda.synchronize()
    return output, [input_.grad for input_ in inputs_on_cuda if input_.requires_grad is True]

###################################################################################

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
            #self.bias.zero_()
            self.bias.uniform_()

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
    torch.cuda.synchronize()
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

# MegatronGFunction is tested in distributed test files
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
            #self.bias.zero_()
            self.bias.uniform_()

    def forward(self, model_input):
        model_input = model_input + self.bias
        out = self.copy_(model_input)
        return out

def test_MegatronF():
    output_size = 1024
    m = MegatronFModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])
    torch.cuda.synchronize()
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


class ScalarAndTupleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, beta, gamma):
        print("ScalarAndTupleFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.gamma = gamma
        return alpha * beta[0] * beta[1] * gamma * input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        gamma = ctx.gamma
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return alpha * beta[0] * beta[1] * gamma * grad_input, None, None, None

class ScalarAndTupleModel(torch.nn.Module):
    def __init__(self, output_size):
        super(ScalarAndTupleModel, self).__init__()
        self.activation = ScalarAndTupleFunction.apply
        self.linear_a = torch.nn.Linear(output_size, output_size)
        self.linear_b = torch.nn.Linear(output_size, output_size)

    def forward(self, x):
        h = self.linear_a(x)
        h = self.activation(h, 5.0, (-1.0, 2.0), -1.0)
        h = self.linear_b(h)
        return h

def test_ScalarAndTuple():
    output_size = 2
    m = ScalarAndTupleModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])
    torch.cuda.synchronize()
    class ScalarAndTupleFunctionWrapperModule(torch.nn.Module):
        def __init__(self):
            super(ScalarAndTupleFunctionWrapperModule, self).__init__()
            self.input_tensors = []
            self.forward_outputs = []
            self.output_tensor = None

        def compute(self, x, alpha, beta, gamma):
            try:
                import builtins
                self.input_tensors = [from_dlpack(i) for i in [x]]
                # what if custom function modify x, and in ORT is using an unexpected value at the same time.
                # todo: assign requires_grad according to original attributes.
                for i_t in self.input_tensors:
                    i_t.requires_grad = True
                with torch.enable_grad():
                    print("==== Entering ScalarAndTupleFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    self.output_tensor = ScalarAndTupleFunction.apply(*self.input_tensors, alpha, beta, gamma)
                    # print(self.output_tensor)
                    # print("===== MegatronFFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.output_tensor, self.output_tensor.device, self.output_tensor.grad_fn))
                    forward_outputs = [self.output_tensor] #[ret.contiguous()]
                    [print("===== ScalarAndTupleFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                    # need hold the forward outputs before PythonOp Compute completed.
                    self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                    [print("===== ScalarAndTupleFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                    ctx_ptr = int(id(self.output_tensor.grad_fn))
                    # ctx_ptr = int(id(ret))
                    #print(self.y.grad_fn.saved_tensors)
                    return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                    print(return_vals)
                    print("==== Exiting ScalarAndTupleFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    return tuple(return_vals)
            except Exception as e:
                print("ScalarAndTupleFunctionWrapperModule:", e)
                return []

        def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = ScalarAndTupleFunction.backward(ctx, self.x_t)
                forward_outputs = list(ret) #[ret.contiguous()] 

                [print("ScalarAndTupleFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("ScalarAndTupleFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("ScalarAndTupleFunctionWrapperModule backward_compute:", e)
                return []

    ort.register_custom_torch_function_forward("ScalarAndTupleFunction", ScalarAndTupleFunctionWrapperModule)
    ort.register_custom_torch_function_backward("ScalarAndTupleFunction", ScalarAndTupleFunctionWrapperModule)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

#############################################################

class InplaceUpdateInputAsOutputNotRequireGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputAsOutputNotRequireGradFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        return inplace_update_input.add_(3 * bias)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class InplaceUpdateInputAsOutputNotRequireGradModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputAsOutputNotRequireGradModel, self).__init__()
        self.inplace_op = InplaceUpdateInputAsOutputNotRequireGradFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            #self.bias.zero_()
            self.bias.uniform_()

    def forward(self, model_input):
        x = model_input.mul(2)
        y1 = self.inplace_op(self.bias, x) # x did not require grad
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

def test_InplaceUpdateInputAsOutputNotRequireGrad():
    output_size = 1024
    m = InplaceUpdateInputAsOutputNotRequireGradModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])
    torch.cuda.synchronize()
    class InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule(torch.nn.Module):
        def __init__(self):
            super(InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule, self).__init__()
            self.input_tensors = []
            self.forward_outputs = []
            self.output_tensor = None

        def compute(self, x, y):
            try:
                import builtins
                self.input_tensors = [from_dlpack(i) for i in [x, y]]
                # what if custom function modify x, and in ORT is using an unexpected value at the same time.
                # todo: assign requires_grad according to original attributes.
                self.input_tensors[0].requires_grad = True
                with torch.enable_grad():
                    print("==== Entering InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    self.output_tensor = InplaceUpdateInputAsOutputNotRequireGradFunction.apply(*self.input_tensors)
                    # print(self.output_tensor)
                    # print("===== MegatronFFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.output_tensor, self.output_tensor.device, self.output_tensor.grad_fn))
                    forward_outputs = [self.output_tensor] #[ret.contiguous()]
                    [print("===== InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                    # need hold the forward outputs before PythonOp Compute completed.
                    self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                    [print("===== InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                    ctx_ptr = int(id(self.output_tensor.grad_fn))
                    # ctx_ptr = int(id(ret))
                    #print(self.y.grad_fn.saved_tensors)
                    return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                    print(return_vals)
                    print("==== Exiting InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule:", e)
                return []

        def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = InplaceUpdateInputAsOutputNotRequireGradFunction.backward(ctx, self.x_t)
                forward_outputs = list(ret) #[ret.contiguous()] 

                [print("InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule backward_compute:", e)
                return []

    ort.register_custom_torch_function_forward("InplaceUpdateInputAsOutputNotRequireGradFunction", InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule)
    ort.register_custom_torch_function_backward("InplaceUpdateInputAsOutputNotRequireGradFunction", InplaceUpdateInputAsOutputNotRequireGradFunctionWrapperModule)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

#########################################################################################

class InplaceUpdateInputNotAsOutputNotRequireGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputNotAsOutputNotRequireGradFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        inplace_update_input.add_(3 * bias)
        return inplace_update_input * 5

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class InplaceUpdateInputNotAsOutputNotRequireGradModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputNotAsOutputNotRequireGradModel, self).__init__()
        self.inplace_op = InplaceUpdateInputNotAsOutputNotRequireGradFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            #self.bias.zero_()
            self.bias.uniform_()

    def forward(self, model_input):
        x = model_input.mul(2)
        y1 = self.inplace_op(self.bias, x)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

def test_InplaceUpdateInputNotAsOutputNotRequireGrad():
    output_size = 1024
    m = InplaceUpdateInputNotAsOutputNotRequireGradModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])
    torch.cuda.synchronize()
    class InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule(torch.nn.Module):
        def __init__(self):
            super(InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule, self).__init__()
            self.input_tensors = []
            self.forward_outputs = []
            self.output_tensor = None

        def compute(self, x, y):
            try:
                import builtins
                self.input_tensors = [from_dlpack(i) for i in [x, y]]
                # what if custom function modify x, and in ORT is using an unexpected value at the same time.
                # todo: assign requires_grad according to original attributes.
                self.input_tensors[0].requires_grad = True
                with torch.enable_grad():
                    print("==== Entering InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    self.output_tensor = InplaceUpdateInputNotAsOutputNotRequireGradFunction.apply(*self.input_tensors)
                    # print(self.output_tensor)
                    # print("===== MegatronFFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.output_tensor, self.output_tensor.device, self.output_tensor.grad_fn))
                    forward_outputs = [self.output_tensor] #[ret.contiguous()]
                    [print("===== InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                    # need hold the forward outputs before PythonOp Compute completed.
                    self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                    [print("===== InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                    ctx_ptr = int(id(self.output_tensor.grad_fn))
                    # ctx_ptr = int(id(ret))
                    #print(self.y.grad_fn.saved_tensors)
                    return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                    print(return_vals)
                    print("==== Exiting InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule:", e)
                return []

        def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = InplaceUpdateInputNotAsOutputNotRequireGradFunction.backward(ctx, self.x_t)
                forward_outputs = list(ret) #[ret.contiguous()] 

                [print("InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule backward_compute:", e)
                return []

    ort.register_custom_torch_function_forward("InplaceUpdateInputNotAsOutputNotRequireGradFunction", InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule)
    ort.register_custom_torch_function_backward("InplaceUpdateInputNotAsOutputNotRequireGradFunction", InplaceUpdateInputNotAsOutputNotRequireGradFunctionWrapperModule)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

#########################################################################################

class InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        ctx.mark_dirty(inplace_update_input)
        # Be noted: if we make the input dirty, we must also put the input in outputs, otherwise, we will get such an error:
        # "RuntimeError: Some elements marked as dirty during the forward method were not returned as output. The inputs that are modified inplace must all be outputs of the Function.""
        return inplace_update_input.add_(3 * bias)

    @staticmethod
    def backward(ctx, grad_output):
        # Bypass the reduce if we are using only 1 GPU.
        return grad_output, None

class InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyModel, self).__init__()
        self.inplace_op = InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            # self.bias.zero_()
            self.bias.uniform_()
            print(self.bias)

    def forward(self, model_input):
        x = model_input.mul(2)
        y1 = self.inplace_op(self.bias, x)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

#                       model_input
#                           |
#                         Mul(2)
#                           |
#                      PythonOP (inplace update)
#                          /   \
#                         /     \
#                      Add       Add
#                        \       /
#                         \     /
#                           Add
#                            |
#                          output0
def test_InplaceUpdateInputAsOutputNotRequireGradWithMarkDirty():
    output_size = 1024
    m = InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    class InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule(torch.nn.Module):
        def __init__(self):
            super(InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule, self).__init__()
            self.input_tensors = []
            self.forward_outputs = []
            self.output_tensor = None

        def compute(self, x, y):
            try:
                import builtins
                self.input_tensors = [from_dlpack(i) for i in [x, y]]
                # what if custom function modify x, and in ORT is using an unexpected value at the same time.
                # todo: assign requires_grad according to original attributes.
                self.input_tensors[0].requires_grad = True
                address_for_torch_tensor = int(id(self.input_tensors[1]))
                with torch.enable_grad():
                    print("==== Entering InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    self.output_tensor = InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction.apply(*self.input_tensors)
                    # print(self.output_tensor)
                    # print("===== MegatronFFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.output_tensor, self.output_tensor.device, self.output_tensor.grad_fn))
                    forward_outputs = [self.output_tensor] #[ret.contiguous()]
                    [print("===== InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                    address_for_output_torch_tensor = int(id(forward_outputs[0]))
                    if address_for_output_torch_tensor != address_for_torch_tensor:
                        raise ValueError("The output torch tensor should reuse the input torch tensor, but actually not.")

                    # need hold the forward outputs before PythonOp Compute completed.
                    self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                    [print("===== InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                    ctx_ptr = int(id(self.output_tensor.grad_fn))
                    # ctx_ptr = int(id(ret))
                    #print(self.y.grad_fn.saved_tensors)
                    return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                    print(return_vals)
                    print("==== Exiting InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule:", e)
                return []

        def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction.backward(ctx, self.x_t)
                forward_outputs = list(ret) #[ret.contiguous()] 

                [print("InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule backward_compute:", e)
                return []

    ort.register_custom_torch_function_forward("InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction", InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule)
    ort.register_custom_torch_function_backward("InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction", InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunctionWrapperModule)
    print("input data: ", x)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    print("comparing forward outputs")
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    print("comparing gradient outputs")
    compare_numpy_list(val_a, val_b)


#########################################################################################

class InplaceUpdateInputAsOutputRequireGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputAsOutputRequireGradFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        # Be noted: if we make the input dirty, we must also put the input in outputs, otherwise, we will get such an error:
        # "RuntimeError: Some elements marked as dirty during the forward method were not returned as output. The inputs that are modified inplace must all be outputs of the Function.""
        return inplace_update_input.add_(3 * bias)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class InplaceUpdateInputAsOutputRequireGradModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputAsOutputRequireGradModel, self).__init__()
        self.inplace_op = InplaceUpdateInputAsOutputRequireGradFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            # self.bias.zero_()
            self.bias.uniform_()
            print(self.bias)

    def forward(self, model_input):
        x = model_input + self.bias
        y1 = self.inplace_op(self.bias, x)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

#                       model_input
#                           |
#                         Mul(2)
#                           |
#                      PythonOP (inplace update)
#                          /   \
#                         /     \
#                      Add       Add
#                        \       /
#                         \     /
#                           Add
#                            |
#                          output0
# This case is known to have an warning message: "The output torch tensor @140214094625024, 140212816617984 should reuse the input torch tensor @140214095996104, 140212816617984 but actually not."
# So seems, if we don't have mark_dirty() in auto grad forward, the result is not using the input_, (maybe a view of it, because data address is same)
def test_InplaceUpdateInputAsOutputRequireGrad():
    output_size = 1024
    m = InplaceUpdateInputAsOutputRequireGradModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    class InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule(torch.nn.Module):
        def __init__(self):
            super(InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule, self).__init__()
            self.input_tensors = []
            self.forward_outputs = []
            self.output_tensor = None

        def compute(self, x, y):
            try:
                import builtins
                raw_input_tensors = [from_dlpack(i) for i in [x, y]]
                # what if custom function modify x, and in ORT is using an unexpected value at the same time.
                # todo: assign requires_grad according to original attributes.

                # the first input are going to do in-place update.
                # we clone a copy for it, .clone() produces a new tensor instance with a new memory allocation to the tensor data.
                # so the in-place operation of the auto grad funtion on the first parameter, will be done on the cloned tensor.
                # and that tensor will be a the forward output wrapped with DLPack ORTValue and return back to ORT C++ kernel.
                self.input_tensors.append(raw_input_tensors[0])
                self.input_tensors.append(raw_input_tensors[1].detach().clone())
                self.input_tensors[0].requires_grad = True
                self.input_tensors[1].requires_grad = True
                print(self.input_tensors[1].is_leaf)
                with torch.enable_grad():
                    adujsted_input_tensor=[self.input_tensors[0], self.input_tensors[1].view(list(self.input_tensors[1].shape))]
                    address_for_torch_tensor = int(id(adujsted_input_tensor[1]))
                    address_for_torch_tensor_data = adujsted_input_tensor[1].data_ptr()
                    print(adujsted_input_tensor[1].is_leaf)
                    print("==== Entering InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    self.output_tensor = InplaceUpdateInputAsOutputRequireGradFunction.apply(*adujsted_input_tensor)
                    # print(self.output_tensor)
                    # print("===== MegatronFFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.output_tensor, self.output_tensor.device, self.output_tensor.grad_fn))
                    forward_outputs = [self.output_tensor] #[ret.contiguous()]
                    [print("===== InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                    address_for_output_torch_tensor = int(id(forward_outputs[0]))
                    if address_for_output_torch_tensor != address_for_torch_tensor:
                        print("WARINING: The output torch tensor @{}, {} should reuse the input torch tensor @{}, {} but actually not.".format(address_for_output_torch_tensor, address_for_torch_tensor_data, address_for_torch_tensor, forward_outputs[0].data_ptr()))

                    # need hold the forward outputs before PythonOp Compute completed.
                    self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                    [print("===== InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                    ctx_ptr = int(id(self.output_tensor.grad_fn))
                    # ctx_ptr = int(id(ret))
                    #print(self.y.grad_fn.saved_tensors)
                    return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                    print(return_vals)
                    print("==== Exiting InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule:", e)
                return []

        def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = InplaceUpdateInputAsOutputRequireGradFunction.backward(ctx, self.x_t)
                forward_outputs = list(ret) #[ret.contiguous()] 

                [print("InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule backward_compute:", e)
                return []

    ort.register_custom_torch_function_forward("InplaceUpdateInputAsOutputRequireGradFunction", InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule)
    ort.register_custom_torch_function_backward("InplaceUpdateInputAsOutputRequireGradFunction", InplaceUpdateInputAsOutputRequireGradFunctionWrapperModule)
    print("input data: ", x)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    print("comparing forward outputs")
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    print("comparing gradient outputs")
    compare_numpy_list(val_a, val_b)


#########################################################################################

class InplaceUpdateInputNotAsOutputRequireGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputNotAsOutputRequireGradFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        inplace_update_input.add_(3 * bias)
        return inplace_update_input * 5

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class InplaceUpdateInputNotAsOutputRequireGradModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputNotAsOutputRequireGradModel, self).__init__()
        self.inplace_op = InplaceUpdateInputNotAsOutputRequireGradFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            # self.bias.zero_()
            self.bias.uniform_()
            print(self.bias)

    def forward(self, model_input):
        x = model_input + self.bias
        y1 = self.inplace_op(self.bias, x)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

#                       model_input
#                           |
#                         Mul(2)
#                           |
#                      PythonOP (inplace update)
#                          /   \
#                         /     \
#                      Add       Add
#                        \       /
#                         \     /
#                           Add
#                            |
#                          output0
# This case is known to have an warning message: "The output torch tensor @140214094625024, 140212816617984 should reuse the input torch tensor @140214095996104, 140212816617984 but actually not."
# So seems, if we don't have mark_dirty() in auto grad forward, the result is not using the input_, (maybe a view of it, because data address is same)
def test_InplaceUpdateInputNotAsOutputRequireGrad():
    output_size = 1024
    m = InplaceUpdateInputNotAsOutputRequireGradModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    class InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule(torch.nn.Module):
        def __init__(self):
            super(InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule, self).__init__()
            self.input_tensors = []
            self.forward_outputs = []
            self.output_tensor = None

        def compute(self, x, y):
            try:
                import builtins
                raw_input_tensors = [from_dlpack(i) for i in [x, y]]
                # what if custom function modify x, and in ORT is using an unexpected value at the same time.
                # todo: assign requires_grad according to original attributes.

                # the first input are going to do in-place update.
                # we clone a copy for it, .clone() produces a new tensor instance with a new memory allocation to the tensor data.
                # so the in-place operation of the auto grad funtion on the first parameter, will be done on the cloned tensor.
                # and that tensor will be a the forward output wrapped with DLPack ORTValue and return back to ORT C++ kernel.
                self.input_tensors.append(raw_input_tensors[0])
                self.input_tensors.append(raw_input_tensors[1].detach().clone())
                self.input_tensors[0].requires_grad = True
                self.input_tensors[1].requires_grad = True
                print(self.input_tensors[1].is_leaf)
                with torch.enable_grad():
                    adujsted_input_tensor=[self.input_tensors[0], self.input_tensors[1].view(list(self.input_tensors[1].shape))]
                    address_for_torch_tensor = int(id(adujsted_input_tensor[1]))
                    address_for_torch_tensor_data = adujsted_input_tensor[1].data_ptr()
                    print(adujsted_input_tensor[1].is_leaf)
                    print("==== Entering InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    self.output_tensor = InplaceUpdateInputNotAsOutputRequireGradFunction.apply(*adujsted_input_tensor)
                    # print(self.output_tensor)
                    # print("===== MegatronFFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.output_tensor, self.output_tensor.device, self.output_tensor.grad_fn))
                    forward_outputs = [self.output_tensor] #[ret.contiguous()]
                    [print("===== InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                    # address_for_output_torch_tensor = int(id(forward_outputs[0]))
                    # if address_for_output_torch_tensor != address_for_torch_tensor:
                    #     print("WARINING: The output torch tensor @{}, {} should reuse the input torch tensor @{}, {} but actually not.".format(address_for_output_torch_tensor, address_for_torch_tensor_data, address_for_torch_tensor, forward_outputs[0].data_ptr()))

                    # need hold the forward outputs before PythonOp Compute completed.
                    self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                    [print("===== InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                    ctx_ptr = int(id(self.output_tensor.grad_fn))
                    # ctx_ptr = int(id(ret))
                    #print(self.y.grad_fn.saved_tensors)
                    return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                    print(return_vals)
                    print("==== Exiting InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule:", e)
                return []

        def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = InplaceUpdateInputNotAsOutputRequireGradFunction.backward(ctx, self.x_t)
                forward_outputs = list(ret) #[ret.contiguous()] 

                [print("InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule backward_compute:", e)
                return []

    ort.register_custom_torch_function_forward("InplaceUpdateInputNotAsOutputRequireGradFunction", InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule)
    ort.register_custom_torch_function_backward("InplaceUpdateInputNotAsOutputRequireGradFunction", InplaceUpdateInputNotAsOutputRequireGradFunctionWrapperModule)
    print("input data: ", x)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    print("comparing forward outputs")
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    print("comparing gradient outputs")
    compare_numpy_list(val_a, val_b)


#########################################################################################

class InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, inplace_update_input):
        print("InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(inplace_update_input, bias)
        ctx.mark_dirty(inplace_update_input)
        # Be noted: if we make the input dirty, we must also put the input in outputs, otherwise, we will get such an error:
        # "RuntimeError: Some elements marked as dirty during the forward method were not returned as output. The inputs that are modified inplace must all be outputs of the Function.""
        return inplace_update_input.add_(3 * bias)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class InplaceUpdateInputAsOutputRequireGradWithMarkDirtyModel(torch.nn.Module):
    def __init__(self, output_size):
        super(InplaceUpdateInputAsOutputRequireGradWithMarkDirtyModel, self).__init__()
        self.inplace_op = InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            # self.bias.zero_()
            self.bias.uniform_()
            print(self.bias)

    def forward(self, model_input):
        x = model_input + self.bias
        y1 = self.inplace_op(self.bias, x)
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out

#                       model_input
#                           |
#                         Mul(2)
#                           |
#                      PythonOP (inplace update)
#                          /   \
#                         /     \
#                      Add       Add
#                        \       /
#                         \     /
#                           Add
#                            |
#                          output0
def test_InplaceUpdateInputAsOutputRequireGradWithMarkDirty():
    output_size = 1024
    m = InplaceUpdateInputAsOutputRequireGradWithMarkDirtyModel(output_size)
    x = torch.randn(output_size, dtype=torch.float)
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    class InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule(torch.nn.Module):
        def __init__(self):
            super(InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule, self).__init__()
            self.input_tensors = []
            self.forward_outputs = []
            self.output_tensor = None

        def compute(self, x, y):
            try:
                import builtins
                raw_input_tensors = [from_dlpack(i) for i in [x, y]]
                # what if custom function modify x, and in ORT is using an unexpected value at the same time.
                # todo: assign requires_grad according to original attributes.

                # the first input are going to do in-place update.
                # we clone a copy for it, .clone() produces a new tensor instance with a new memory allocation to the tensor data.
                # so the in-place operation of the auto grad cuntion on the first parameter, will be done on the cloned tensor.
                # and that tensor will be a the forward output wrapped with DLPack ORTValue and return back to ORT C++ kernel.
                self.input_tensors.append(raw_input_tensors[0])
                self.input_tensors.append(raw_input_tensors[1].detach().clone())
                self.input_tensors[0].requires_grad = True
                self.input_tensors[1].requires_grad = True
                print(self.input_tensors[1].is_leaf)
                with torch.enable_grad():
                    adujsted_input_tensor=[self.input_tensors[0], self.input_tensors[1].view(list(self.input_tensors[1].shape))]
                    address_for_torch_tensor = int(id(adujsted_input_tensor[1]))
                    address_for_torch_tensor_data = adujsted_input_tensor[1].data_ptr()
                    print(adujsted_input_tensor[1].is_leaf)
                    print("==== Entering InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    self.output_tensor = InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction.apply(*adujsted_input_tensor)
                    # print(self.output_tensor)
                    # print("===== MegatronFFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.output_tensor, self.output_tensor.device, self.output_tensor.grad_fn))
                    forward_outputs = [self.output_tensor] #[ret.contiguous()]
                    [print("===== InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                    address_for_output_torch_tensor = int(id(forward_outputs[0]))
                    if address_for_output_torch_tensor != address_for_torch_tensor:
                        raise ValueError("The output torch tensor should reuse the input torch tensor, but actually not.")

                    # need hold the forward outputs before PythonOp Compute completed.
                    self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                    [print("===== InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                    ctx_ptr = int(id(self.output_tensor.grad_fn))
                    # ctx_ptr = int(id(ret))
                    #print(self.y.grad_fn.saved_tensors)
                    return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                    print(return_vals)
                    print("==== Exiting InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                    return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule:", e)
                return []

        def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction.backward(ctx, self.x_t)
                forward_outputs = list(ret) #[ret.contiguous()] 

                [print("InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule backward_compute:", e)
                return []

    ort.register_custom_torch_function_forward("InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction", InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule)
    ort.register_custom_torch_function_backward("InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction", InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunctionWrapperModule)
    print("input data: ", x)
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x], [output_size])

    torch.cuda.synchronize()
    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    print("comparing forward outputs")
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    print("comparing gradient outputs")
    compare_numpy_list(val_a, val_b)

test_GeLU()
test_MegatronF()
test_ScalarAndTuple()

## test case, some input are in-place updated, and the input did not require gradient.
test_InplaceUpdateInputAsOutputNotRequireGrad()
test_InplaceUpdateInputNotAsOutputNotRequireGrad()
test_InplaceUpdateInputAsOutputNotRequireGradWithMarkDirty()

### test case, some input are in-place updated, and the input require gradient.
test_InplaceUpdateInputAsOutputRequireGrad()
test_InplaceUpdateInputNotAsOutputRequireGrad()
test_InplaceUpdateInputAsOutputRequireGradWithMarkDirty()
