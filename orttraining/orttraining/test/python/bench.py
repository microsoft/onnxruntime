# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Import external libraries.
from functools import wraps
import onnxruntime
import pytest
import torch
from torch.nn.parameter import Parameter

# Import ORT modules.
from test_helper_utils import *
from onnxruntime.training.ortmodule import ORTModule

import argparse
from typing import Text
import nvtx
import psutil

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int)
parser.add_argument("--hidden", type=int)
parser.add_argument("--layer", type=int)
parser.add_argument('--ort', action='store_true')
parser.add_argument("--tag", type=str)

args= parser.parse_args()

torch.manual_seed(1)
onnxruntime.set_seed(1)

import psutil
def mem_stat(name):
    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    print(f"=========== {name} MA {round(torch.cuda.memory_allocated() / (1024 * 1024),4 )} MB \
        Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024),4)} MB \
        CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%")
        # CA {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        # Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

def test_CustomFunctionOverheadTest(b, h, run_with_ort):
    class CustomFunctionOverheadTestFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight):
            mem_stat("enter forward!!!!!!!!!!!!!!!!!!!!")
            with nvtx.annotate("forward", color="green"):
                # val = input.clamp(min=0)
                #new_val=weight
                #for i in range(1):
                new_val = torch.matmul(input, weight)
                # ctx.save_for_backward(input)
                # res2 = input *2
                mem_stat("exiting forward!!!!!!!!!!!!!!!!!!!!")
                return new_val #, res2

        @staticmethod
        def backward(ctx, grad_output):
            mem_stat("enter backward!!!!!!!!!!!!!!!!!!!!")
            with nvtx.annotate("backward", color="green"):
                # input, = ctx.saved_tensors
                res1 = grad_output * grad_output
                # mem_stat("after first command")
                # res1 = grad_output
                trans = torch.transpose(grad_output, 1, 0)
                # mem_stat("after second - transpose command")
                trans2 = trans.contiguous()
                # mem_stat("after third - trans2 command")
                res2 = torch.matmul(trans2, grad_output)
                mem_stat("exiting backward!!!!!!!!!!!!!!!!!!!!")
                return res1 , res2


    class CustomFunctionDelegation(torch.nn.Module):
        def __init__(self, output_size):
            super(CustomFunctionDelegation, self).__init__()
            self._p = Parameter(torch.empty(
                (output_size, output_size),
                device=torch.cuda.current_device(),
                dtype=torch.float))

            with torch.no_grad():
                self._p.uniform_()
            self.custom_func = CustomFunctionOverheadTestFunction.apply

        def forward(self, x):
            # print("self._p.requires_grad()", self._p.requires_grad)
            return self.custom_func(x, self._p)
            # return torch.matmul(x, self._p)

    class CustomFunctionOverheadTestModel(torch.nn.Module):
        def __init__(self, output_size):
            super(CustomFunctionOverheadTestModel, self).__init__()
            self._layer_count = args.layer
            self._layers = torch.nn.ModuleList(
                [CustomFunctionDelegation(output_size) for i in range(self._layer_count)])


        def forward(self, x):
            for index, val in enumerate(self._layers):
                x = self._layers[index](x)
            # mem_stat("before x * 9")
            # # x = x * 9
            # mem_stat("after x * 9")
            return x

    output_size = h
    batch_size = b
    def model_builder():
        return CustomFunctionOverheadTestModel(output_size)

    def input_generator():
        return torch.randn(batch_size, output_size, dtype=torch.float).requires_grad_()

    # generate a label that have same shape as forward output.
    label_input = torch.ones([batch_size * output_size]).reshape(batch_size, output_size).contiguous()

    def cuda_barrier_func():
        torch.cuda.synchronize()
    cuda = torch.device('cuda:0')

    for i in range(1):
        m = model_builder()
        x = input_generator()

        if run_with_ort:
            #with nvtx.annotate("run_with_ort_on_device", color="red"):
            #outputs_ort, grads_ort, start, end = 
            run_with_ort_on_device(
                cuda, m, [x], label_input)
            cuda_barrier_func()
            #print(args.tag, ", ort run elapsed time (ms): ", start.elapsed_time(end))
        else:
            #with nvtx.annotate("run_with_pytorch_on_device", color="red"):
            #outputs, grads, start, end = 
            run_with_pytorch_on_device(cuda, m, [x], label_input)
            cuda_barrier_func()
            # print(args.tag, ", pt run elapsed time (ms): ", start.elapsed_time(end))


test_CustomFunctionOverheadTest(args.batch, args.hidden, args.ort)


