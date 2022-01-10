import torch
import lazy_tensor_core
#import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as ltm

from onnxruntime.capi.onnxruntime_pybind11_state import register_ort_as_torch_jit_executor
register_ort_as_torch_jit_executor()

torch._C._jit_set_profiling_executor(False)
torch._C._set_graph_executor_optimize(False)
lazy_tensor_core._LAZYC._ltc_init_ts_backend()

torch.manual_seed(42)

device = 'lazy'
dtype = torch.float32

def model(x):
    y = x.relu()
    return y
    #z = y.sum()
    #return z
    #w = z.exp()
    #print('x:\n', x)
    #print('y:\n', y)
    #print('w:\n', w)
    #return w

def run(tag, x):
    print(f'Round {tag}:')
    x = torch.tensor(x, device=device, dtype=dtype).requires_grad_()
    print('F:')
    y = model(x)
    ltm.mark_step()
    print('x:')
    print(x)
    print('y:')
    print(y)

    print('B:')
    y.backward()
    ltm.mark_step()
    print(f'x.grad: {x.grad}')

run(1, 1)
run(2, -1)

#print(metrics.metrics_report())
