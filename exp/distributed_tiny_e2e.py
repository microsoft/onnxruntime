import onnx
import torch
torch.manual_seed(1)
from onnxruntime.training import ORTModule
import onnxruntime as ort
import os
import sys
from torch.utils.dlpack import from_dlpack, to_dlpack
 
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C

from torch import optim
import torch.nn.functional as F

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
        print(ctx.saved_tensors)
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

    def backward_compute(self, ctx, x):
        print(ctx, ctx.saved_tensors)
        self.x_t = from_dlpack(x)
        # this should be False
        #self.x_t.requires_grad = False
        ret = MyReLU.backward(ctx, self.x_t)
        forward_outputs = [ret] #[ret.contiguous()]
        [print("CustomFnWrapperModule.backward_compute: shape: ", a.shape) for a in forward_outputs]
        # need hold the forward outputs before PythonOp Compute completed.
        self.forward_outputs = [_ortvalue_from_dlpack(to_dlpack(r)) for r in forward_outputs]
        [print("CustomFnWrapperModule.backward_compute: tensor->MutableDataRaw addr", int(r.data_ptr())) for r in self.forward_outputs]

        return_vals =[int(r.ortvalue_ptr()) for r in self.forward_outputs]
        print(return_vals)
        return tuple(return_vals)


import torch.distributed as dist
from torch.multiprocessing import Process

data = [torch.tensor([[0.3971, 0.7544],
                    [0.5695, 0.4388]], requires_grad=False),
        torch.tensor([[0.2971, 0.6544],
                    [0.4695, 0.3388]], requires_grad=False),]

def run(rank, size):
    device = torch.device('cuda:' + str(rank))
    # Define input.
    
    # Fake data partition.
    x = data[rank].to(device)
    # x.requires_grad = True

    model = MyModule().to(device)
    torch.onnx.export(model, (x,), 'model_rank_'+str(rank)+'.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, custom_opsets={"prim":1})
    # onnx_model = onnx.load('model.onnx')
    # print(onnx_model)
    # print('-------------------------------------------')
    # print('-------------------------------------------')
    print('Use ORTModule')
    model = ORTModule(model)

    ort.register_custom_torch_function_forward("MyReLU", CustomFnWrapperModule)
    ort.register_custom_torch_function_backward("MyReLU", CustomFnWrapperModule)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    optimizer.zero_grad()
    y = model(x)
    loss = y.sum()
    loss.backward()

    size = float(dist.get_world_size())
    for param in model.parameters():
        print(param.grad.data)
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

    optimizer.step()
    
    print('[', rank, '] x:\n', x)
    print('[', rank, '] x.grad:\n', x.grad)
    print('[', rank, '] y:\n', y)


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', init_method='tcp://' + os.environ['MASTER_ADDR'] + ':23456',
                            world_size=size, rank=rank)
    fn(rank, size)

import torch.multiprocessing as mp

if __name__ == "__main__":
    size = 2
    mp.spawn(init_process, nprocs=size, args=(size, run))