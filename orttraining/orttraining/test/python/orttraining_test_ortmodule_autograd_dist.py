# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import copy
import onnxruntime
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from onnxruntime.training.ortmodule import ORTModule
from onnxruntime.training.ortmodule._graph_execution_manager_factory import GraphExecutionManagerFactory
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parameter import Parameter

import _test_helpers

torch.manual_seed(1)
onnxruntime.set_seed(1)


class ReduceWithMarkDirtyFunction(torch.autograd.Function):
    # All-reduce the input from the model parallel region.
    @staticmethod
    def forward(ctx, arg):
        def reduce(buffer):
            # All-reduce.
            address_for_torch_tensor = int(id(buffer))
            torch.distributed.all_reduce(buffer)
            address_for_output_torch_tensor = int(id(buffer))
            if address_for_output_torch_tensor != address_for_torch_tensor:
                raise ValueError(
                    "The output torch tensor should reuse the input torch tensor, but actually not.")
            return buffer
        ctx.save_for_backward(arg)
        ctx.mark_dirty(arg)
        return reduce(arg)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.add(1.0)


class ReduceWithMarkDirtyModel(torch.nn.Module):
    def __init__(self, dim):
        super(ReduceWithMarkDirtyModel, self).__init__()
        self.reduce_op_ = ReduceWithMarkDirtyFunction.apply
        self.bias = Parameter(torch.empty(
            dim,
            device=torch.cuda.current_device(),
            dtype=torch.float))

        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.uniform_()

    def forward(self, model_input):
        x = model_input + self.bias
        y1 = self.reduce_op_(x)  # at this point x require_grad = True
        y2 = x.add(self.bias)
        out = x + y1 + y2
        return out


def test_Distributed_ReduceWithMarkDirtyModel(rank, size):
    def run_with_pytorch_on_gpu(model, args, rank, device):
        model.to(device)
        cuda_args = [input_.to(device) for input_ in args]
        model = DDP(model, device_ids=[rank])
        output = model(*cuda_args)
        output.sum().backward()
        return output, [arg.grad for arg in cuda_args]


    def run_with_ort_on_gpu(model, args, rank, device):
        model.to(device)
        model = ORTModule(model)

        _test_helpers.set_onnx_fallthrough_export_type(model)
        model = DDP(model, device_ids=[rank])
        cuda_args = [arg.to(device) for arg in args]
        output = model(*cuda_args)
        output.sum().backward()
        return output, [arg.grad for arg in cuda_args]

    try:
        torch.cuda.set_device('cuda:' + str(rank))
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend='nccl', init_method='tcp://' + os.environ['MASTER_ADDR'] + ':23456',
                                world_size=size, rank=rank)

        dim = 32
        device = torch.device('cuda:' + str(rank))
        x = torch.randn(dim, dtype=torch.float)
        x.requires_grad = True
        x_copy = copy.deepcopy(x)
        m = ReduceWithMarkDirtyModel(dim)

        torch.cuda.synchronize()

        outputs, grads = run_with_pytorch_on_gpu(
            m, [x], rank, device)

        torch.cuda.synchronize()

        outputs_ort, grads_ort = run_with_ort_on_gpu(
            m, [x_copy], rank, device)

        torch.cuda.synchronize()

        val_list_a = [o.detach().cpu() for o in outputs if o is not None]
        val_list_b = [o.detach().cpu() for o in outputs_ort if o is not None]
        _test_helpers.compare_tensor_list(val_list_a, val_list_b)

        val_list_a = [o.detach().cpu() for o in grads if o is not None]
        val_list_b = [o.detach().cpu() for o in grads_ort if o is not None]
        _test_helpers.compare_tensor_list(val_list_a, val_list_b)
    except Exception as e:
        print(
            f"test_Distributed_ReduceWithMarkDirtyModel fail with rank {rank} with world size {size} with exception: \n{e}.")
        raise e


if __name__ == "__main__":
    size = 2
    try:
        mp.spawn(test_Distributed_ReduceWithMarkDirtyModel,
                 nprocs=size, args=(size,))
    except:
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        raise
