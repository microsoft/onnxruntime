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

# distributed requirements start
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.multiprocessing import Process
from torch import optim
import torch.nn.functional as F
# distributed requirements end

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

def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    # All-reduce.
    address_for_torch_tensor = int(id(input_))
    torch.distributed.all_reduce(input_)
    address_for_output_torch_tensor = int(id(input_))
    if address_for_output_torch_tensor != address_for_torch_tensor:
        raise ValueError("The output torch tensor should reuse the input torch tensor, but actually not.")
    else:
        print("!!!!!!!!!!!torch.distributed.all_reduce operates inplace")
    return input_


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', init_method='tcp://' + os.environ['MASTER_ADDR'] + ':23456',
                            world_size=size, rank=rank)
    fn(rank, size)

def run_with_pytorch_on_gpu(model, input_list, output_shape, device, optimizer):
    print('Use PyTorch for CUDA run....')
    model.to(device)
    inputs_on_cuda = [input_.to(device) for input_ in input_list]
    output = model(*inputs_on_cuda)
    criterion = torch.nn.MSELoss()

    target=torch.ones(*output_shape).to(device)
    loss = criterion(output, target)
    loss.backward()
    torch.cuda.synchronize()
    return output, [input_.grad for input_ in inputs_on_cuda if input_.requires_grad is True]

def run_with_ort_on_gpu(model, input_list, output_shape, device, optimizer):
    print('Use ORTModule for CUDA run....')
    model.to(device)
    model = ORTModule(model)
    inputs_on_cuda = [input_.to(device) for input_ in input_list]
    output = model(*inputs_on_cuda)
    criterion = torch.nn.MSELoss()

    target=torch.ones(*output_shape).to(device)
    loss = criterion(output, target)
    loss.backward()

    # size = float(dist.get_world_size())
    # for param in model.parameters():
    #     print(param.grad.data)
    #     dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
    #     param.grad.data /= size

    # optimizer.step()
    torch.cuda.synchronize()
    return output, [input_.grad for input_ in inputs_on_cuda if input_.requires_grad is True]


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

#####################################################################################################

class ReduceWithoutMarkDirtyFunction(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        print("ReduceWithoutMarkDirtyFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(input_)
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ReduceWithoutMarkDirtyModel(torch.nn.Module):
    def __init__(self, output_size):
        super(ReduceWithoutMarkDirtyModel, self).__init__()
        self.reduce_op_ = ReduceWithoutMarkDirtyFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()
            # self.bias.uniform_()
            print(self.bias)

    def forward(self, model_input):
        x = model_input + self.bias
        y1 = self.reduce_op_(x)  # at this point x require_grad = True
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out


class ReduceWithoutMarkDirtyFunctionWrapperModule(torch.nn.Module):
    def __init__(self):
        super(ReduceWithoutMarkDirtyFunctionWrapperModule, self).__init__()
        self.x_t = None
        self.forward_outputs = []
        self.y = None

    def compute(self, x):
        try:
            init_x = from_dlpack(x)
            # # reshape won't change the data pointer, but the tensor is changed.
            # # there might be a problem: as init_x is teared down, the underlying ORTValue desctor will also be called.
            # self.x_t = init_x.reshape(list(init_x.shape))
            self.x_t = torch.clone(init_x)
            self.x_t.requires_grad = True
            print(self.x_t.is_leaf)
            with torch.enable_grad():
                input_torch_tensor = self.x_t.view(list(self.x_t.shape))
                print(input_torch_tensor.is_leaf)
                print("==== Entering ReduceWithoutMarkDirtyFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                self.y = ReduceWithoutMarkDirtyFunction.apply(input_torch_tensor)
                # print(self.y)
                # print("===== ReduceWithoutMarkDirtyFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.y, self.y.device, self.y.grad_fn))
                forward_outputs = [self.y] #[ret.contiguous()]
                [print("===== ReduceWithoutMarkDirtyFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                # need hold the forward outputs before PythonOp Compute completed.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                [print("===== ReduceWithoutMarkDirtyFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                ctx_ptr = int(id(self.y.grad_fn))
                # ctx_ptr = int(id(ret))
                #print(self.y.grad_fn.saved_tensors)
                return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                print("==== Exiting ReduceWithoutMarkDirtyFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                return tuple(return_vals)
        except Exception as e:
            print(e)
            return []

    def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = ReduceWithoutMarkDirtyFunction.backward(ctx, self.x_t)
                forward_outputs = [ret] #[ret.contiguous()]  # for single result, please don't use list() to do conversion.

                [print("ReduceWithoutMarkDirtyFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("ReduceWithoutMarkDirtyFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("ReduceWithoutMarkDirtyFunctionWrapperModule backward_compute:", e)
                return []


def test_Distributed_ReduceWithoutMarkDirtyModel(rank):
    output_size = 1024
    device = torch.device('cuda:' + str(rank))
    x = torch.randn(output_size, dtype=torch.float)
    print(x)
    x_copy = copy.deepcopy(x)
    x_for_export = torch.randn(output_size, dtype=torch.float)
    m = ReduceWithoutMarkDirtyModel(output_size)
    torch.onnx.export(copy.deepcopy(m).to(device), (x_for_export.to(device), ), 'model_rank_' + str(rank) + '.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, custom_opsets={"prim":1})

    m = copy.deepcopy(m)
    optimizer = optim.SGD(m.parameters(), lr=0.01, momentum=0.5)
    optimizer.zero_grad()
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size], device, optimizer)
    torch.cuda.synchronize()


    ort.register_custom_torch_function_forward("ReduceWithoutMarkDirtyFunction", ReduceWithoutMarkDirtyFunctionWrapperModule)
    ort.register_custom_torch_function_backward("ReduceWithoutMarkDirtyFunction", ReduceWithoutMarkDirtyFunctionWrapperModule)

    optimizer = optim.SGD(m.parameters(), lr=0.01, momentum=0.5)
    optimizer.zero_grad()
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x_copy], [output_size], device, optimizer)
    torch.cuda.synchronize()

    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)

#####################################################################################################

class ReduceWithMarkDirtyFunction(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        print("ReduceWithMarkDirtyFunction(torch.autograd.Function) forward")
        ctx.save_for_backward(input_)
        ctx.mark_dirty(input_)
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ReduceWithMarkDirtyModel(torch.nn.Module):
    def __init__(self, output_size):
        super(ReduceWithMarkDirtyModel, self).__init__()
        self.reduce_op_ = ReduceWithMarkDirtyFunction.apply
        self.bias = Parameter(torch.empty(
            output_size,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()
            # self.bias.uniform_()
            print(self.bias)

    def forward(self, model_input):
        x = model_input + self.bias
        y1 = self.reduce_op_(x)  # at this point x require_grad = True
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out


class ReduceWithMarkDirtyFunctionWrapperModule(torch.nn.Module):
    def __init__(self):
        super(ReduceWithMarkDirtyFunctionWrapperModule, self).__init__()
        self.x_t = None
        self.forward_outputs = []
        self.y = None

    def compute(self, x):
        try:
            init_x = from_dlpack(x)
            # # reshape won't change the data pointer, but the tensor is changed.
            # # there might be a problem: as init_x is teared down, the underlying ORTValue desctor will also be called.
            # self.x_t = init_x.reshape(list(init_x.shape))
            self.x_t = torch.clone(init_x)
            self.x_t.requires_grad = True
            print(self.x_t.is_leaf)
            with torch.enable_grad():
                input_torch_tensor = self.x_t.view(list(self.x_t.shape))
                print(input_torch_tensor.is_leaf)
                print("==== Entering ReduceWithMarkDirtyFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                self.y = ReduceWithMarkDirtyFunction.apply(input_torch_tensor)
                # print(self.y)
                # print("===== ReduceWithMarkDirtyFunctionWrapperModule.compute forward output: {} on device {}, grad_fn: {}".format(self.y, self.y.device, self.y.grad_fn))
                forward_outputs = [self.y] #[ret.contiguous()]
                [print("===== ReduceWithMarkDirtyFunctionWrapperModule.compute: shape: ", a.shape) for a in forward_outputs]

                # need hold the forward outputs before PythonOp Compute completed.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
                [print("===== ReduceWithMarkDirtyFunctionWrapperModule.compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                ctx_ptr = int(id(self.y.grad_fn))
                # ctx_ptr = int(id(ret))
                #print(self.y.grad_fn.saved_tensors)
                return_vals = [ctx_ptr] + [int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                print("==== Exiting ReduceWithMarkDirtyFunctionWrapperModule.compute , process id {} ====".format(os.getpid()))
                return tuple(return_vals)
        except Exception as e:
            print(e)
            return []

    def backward_compute(self, ctx, x):
            try:
                #print(ctx, ctx.saved_tensors)
                self.x_t = from_dlpack(x)
                self.x_t.requires_grad = False
                ret = ReduceWithMarkDirtyFunction.backward(ctx, self.x_t)
                forward_outputs = [ret] #[ret.contiguous()]  # for single result, please don't use list() to do conversion.

                [print("ReduceWithMarkDirtyFunctionWrapperModule.backward_compute: shape: ", a.shape if a is not None else None) for a in forward_outputs]
                # need hold the forward outputs before PythonOp Compute completed.
                # todo: we should use a python dict here to pass outputs.
                self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs if r is not None]
                [print("ReduceWithMarkDirtyFunctionWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

                return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
                print(return_vals)
                return tuple(return_vals)
            except Exception as e:
                print("ReduceWithMarkDirtyFunctionWrapperModule backward_compute:", e)
                return []


def test_Distributed_ReduceWithMarkDirtyModel(rank):
    output_size = 1024
    device = torch.device('cuda:' + str(rank))
    x = torch.randn(output_size, dtype=torch.float)
    print(x)
    x_copy = copy.deepcopy(x)
    x_for_export = torch.randn(output_size, dtype=torch.float)
    m = ReduceWithMarkDirtyModel(output_size)
    torch.onnx.export(copy.deepcopy(m).to(device), (x_for_export.to(device), ), 'model_rank_' + str(rank) + '.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, custom_opsets={"prim":1})

    m = copy.deepcopy(m)
    optimizer = optim.SGD(m.parameters(), lr=0.01, momentum=0.5)
    optimizer.zero_grad()
    outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size], device, optimizer)
    torch.cuda.synchronize()


    ort.register_custom_torch_function_forward("ReduceWithMarkDirtyFunction", ReduceWithMarkDirtyFunctionWrapperModule)
    ort.register_custom_torch_function_backward("ReduceWithMarkDirtyFunction", ReduceWithMarkDirtyFunctionWrapperModule)

    optimizer = optim.SGD(m.parameters(), lr=0.01, momentum=0.5)
    optimizer.zero_grad()
    outputs_ort, grads_ort = run_with_ort_on_gpu(m, [x_copy], [output_size], device, optimizer)
    torch.cuda.synchronize()

    val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
    val_b = [o.detach().cpu().numpy() for o in outputs_ort if o is not None]
    compare_numpy_list(val_a, val_b)

    val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
    val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
    compare_numpy_list(val_a, val_b)


def run(rank, size):
    # Known issue: test_Distributed_ReduceWithoutMarkDirtyModel failed because, exported ONNX model is not correct, check the onenote for more details. 
    # test_Distributed_ReduceWithoutMarkDirtyModel(rank)
    test_Distributed_ReduceWithMarkDirtyModel(rank)

if __name__ == "__main__":
    size = 2
    mp.spawn(init_process, nprocs=size, args=(size, run))