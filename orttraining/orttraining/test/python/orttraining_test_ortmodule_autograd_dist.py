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

from onnxruntime.training.ortmodule import _utils, ORTModule
import _test_helpers

from torch.nn.parameter import Parameter
import sys
import onnx
import torch
torch.manual_seed(1)
import onnxruntime as ort
import os
from torch.utils.dlpack import from_dlpack, to_dlpack
 
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C
import copy
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

# distributed requirements start
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.multiprocessing import Process
from torch import optim
import torch.nn.functional as F
import threading
# distributed requirements end

def _ortvalue_from_dlpack(dlpack_tensor):
    return OrtValue(C.OrtValue.from_dlpack(dlpack_tensor, False))

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

def run_with_ort_on_gpu(model, input_list, output_shape, rank, optimizer):
    print('Use ORTModule for CUDA run....')
    device = torch.device('cuda:' + str(rank))
    model.to(device)
    model = ORTModule(model, torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)
    model = DDP(model, device_ids=[rank])
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
    torch.cuda.synchronize(device)
    grad_outputs = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_outputs.append(param.grad)
    return output, grad_outputs


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
    
    ort.register_forward_core("ReduceWithoutMarkDirtyFunction", ReduceWithoutMarkDirtyFunction.apply)
    ort.register_backward_core("ReduceWithoutMarkDirtyFunction", ReduceWithoutMarkDirtyFunction.backward)

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
        return grad_output.add(1.0)

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


def test_Distributed_ReduceWithMarkDirtyModel(rank, size):
    try:
        # import faulthandler
        # faulthandler.enable()
        # sys.stdout = open('stdout_rank_' + str(rank), 'w')
        # sys.stderr = open('stderr_rank_' + str(rank), 'w')
        torch.cuda.set_device('cuda:' + str(rank))
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend='nccl', init_method='tcp://' + os.environ['MASTER_ADDR'] + ':23456',
                                world_size=size, rank=rank)

        output_size = 1024
        device = torch.device('cuda:' + str(rank))
        x = torch.randn(output_size, dtype=torch.float)
        print(x)
        x_copy = copy.deepcopy(x)
        # x_for_export = torch.randn(output_size, dtype=torch.float)
        m = ReduceWithMarkDirtyModel(output_size)
        # torch.onnx.export(copy.deepcopy(m).to(device), (x_for_export.to(device), ), 'model_rank_' + str(rank) + '.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, custom_opsets={"prim":1})

        # pytorch_m = copy.deepcopy(m)
        # pytorch_ddp_m = DDP(pytorch_m, device_ids=[rank])
        # optimizer = optim.SGD(pytorch_ddp_m.parameters(), lr=0.01, momentum=0.5)
        # optimizer.zero_grad()
        # outputs, grads = run_with_pytorch_on_gpu(pytorch_ddp_m, [x], [output_size], device, optimizer)
        # torch.cuda.synchronize()

        ort.register_forward_core("ReduceWithMarkDirtyFunction", ReduceWithMarkDirtyFunction.apply)
        ort.register_backward_core("ReduceWithMarkDirtyFunction", ReduceWithMarkDirtyFunction.backward)

        pt_ort_m = copy.deepcopy(m)
        # optimizer = optim.SGD(pt_ort_m.parameters(), lr=0.01, momentum=0.5)
        # optimizer.zero_grad()
        outputs_ort, grads_ort = run_with_ort_on_gpu(pt_ort_m, [x_copy], [output_size], rank, None)
        # [g.add_(5.0) for g in grads_ort]
        torch.cuda.synchronize()
        print("Rank {}:".format(rank), outputs_ort, grads_ort)

        # print("Rank {} test_Distributed_ReduceWithMarkDirtyModel start comparing".format(rank))
        # # val_a = [o.detach().cpu().numpy() for o in outputs if o is not None]
        # print("Rank {} test_Distributed_ReduceWithMarkDirtyModel start comparing 2222 ".format(rank))
        # print("len(outputs_ort): {}, id(outputs_ort): {}, outputs_ort.data_ptr():{}".format(len(outputs_ort), id(outputs_ort), outputs_ort.data_ptr()))
        # # val_b = [outputs_ort.detach().cpu().numpy()]
        # print("Rank {} test_Distributed_ReduceWithMarkDirtyModel start comparing 3333".format(rank))
        # # compare_numpy_list(val_a, val_b)

        # # val_a = [o.detach().cpu().numpy() for o in grads if o is not None]
        # # val_b = [o.detach().cpu().numpy() for o in grads_ort if o is not None]
        # # compare_numpy_list(val_a, val_b)
        # print("Rank {} test_Distributed_ReduceWithMarkDirtyModel UT ended".format(rank))
        # sys.stdout.close()
        return 0
    except Exception as e:
        print("test_Distributed_ReduceWithMarkDirtyModel:", e)
        return 0 






if __name__ == "__main__":
    size = 2
    try:
        mp.spawn(test_Distributed_ReduceWithMarkDirtyModel, nprocs=size, args=(size,))
    except:
        import sys
        sys.stdout.flush()
        sys.stderr.flush()