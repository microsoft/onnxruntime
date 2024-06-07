import torch
from torch import nn
from torch import Tensor
class QuickGELUActivation(nn.Module):
  def forward(self, input: Tensor) -> Tensor:
    return input * torch.sigmoid(1.702 * input)



class TestSplitQuickGeluMulOp(nn.Module):
  def forward(self, input: Tensor) -> Tensor:
    t1, t2 = torch.split(input, input.shape[-1]//2, -1)
    activation = QuickGELUActivation()
    qg_out = activation(t2)
    return t1*qg_out


temp = TestSplitQuickGeluMulOp()
# inp = torch.randn((1024, 1024, 12), device='cuda:0')
# inp = torch.randn((32, 1024, 1024, 12), device='cuda:0')
# inp = torch.randn((32, 1024, 12, 1024), device='cuda:0')
# inp = torch.randn((76, 54, 1368), device='cuda:0')
# out = temp(inp)

from onnxruntime.training.ortmodule import ORTModule
import time

def compare_torch_ort_perf(inp, num_steps=1000):
  model2 = ORTModule(temp)
  ort_temp_out = model2(inp)
  print("Current input shape:", inp.shape)
  torch.cuda.synchronize()
  start = time.time()
  for i in range(num_steps):
    torch_out = temp(inp)


  torch.cuda.synchronize()
  print("Total time torch:", time.time() - start)
  torch.cuda.synchronize()
  start = time.time()
  for i in range(num_steps):
    ort_out = model2(inp)

  torch.cuda.synchronize()
  print("Total time ORT:", time.time() - start)
  comparison = torch.isclose(torch_out, ort_out, rtol=1e-04, atol=1e-05)
  # Check if all elements are close
  all_close = comparison.all().item()
  print("Are all elements close:", all_close)


inp = torch.randn((1024, 1024, 12), device='cuda:0')
compare_torch_ort_perf(inp, 10000)

inp = torch.randn((32, 1024, 1024, 12), device='cuda:0')
compare_torch_ort_perf(inp, 1000)

inp = torch.randn((32, 1024, 12, 1024), device='cuda:0')
compare_torch_ort_perf(inp, 1000)

inp = torch.randn((76, 54, 1368), device='cuda:0')
compare_torch_ort_perf(inp, 10000)
