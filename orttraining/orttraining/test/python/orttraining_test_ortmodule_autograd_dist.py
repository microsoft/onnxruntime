# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import copy
import os
import sys
import onnx
import numpy as np
import threading

import torch
from torch.nn.parameter import Parameter
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch import optim
import torch.nn.functional as F
# distributed requirements start
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
# distributed requirements end

torch.manual_seed(1)

from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C
from onnxruntime.training.ortmodule import ORTModule

import pytest
import _test_helpers

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

def compare_tensor_list(val_list_a, val_list_b):
    for val_a, val_b in zip(val_list_a,val_list_b):
        _test_helpers.assert_values_are_close(val_a, val_b, atol=1e-7, rtol=1e-6)

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
        torch.cuda.set_device('cuda:' + str(rank))
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend='nccl', init_method='tcp://' + os.environ['MASTER_ADDR'] + ':23456',
                                world_size=size, rank=rank)

        output_size = 1024
        device = torch.device('cuda:' + str(rank))
        x = torch.randn(output_size, dtype=torch.float)
        x_copy = copy.deepcopy(x)
        m = ReduceWithMarkDirtyModel(output_size)

        outputs, grads = run_with_pytorch_on_gpu(m, [x], [output_size], rank, None)
        torch.cuda.synchronize()

        pt_ort_m = copy.deepcopy(m)
        outputs_ort, grads_ort = run_with_ort_on_gpu(pt_ort_m, [x_copy], [output_size], rank, None)
        torch.cuda.synchronize()
        print("Rank {}:".format(rank), outputs_ort, grads_ort)

        val_list_a = [o.detach().cpu() for o in outputs if o is not None]
        val_list_b = [o.detach().cpu() for o in outputs_ort if o is not None]
        compare_tensor_list(val_list_a, val_list_b)

        val_list_a = [o.detach().cpu() for o in grads if o is not None]
        val_list_b = [o.detach().cpu() for o in grads_ort if o is not None]
        compare_tensor_list(val_list_a, val_list_b)
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