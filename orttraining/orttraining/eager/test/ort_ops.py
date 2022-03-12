# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import torch
import onnxruntime_pybind11_state as torch_ort
import numpy as np

class OrtOpTests(unittest.TestCase):
  def get_device(self):
    return torch_ort.device()

  def test_add(self):
    device = self.get_device()
    cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    ort_ones = cpu_ones.to(device)
    cpu_twos = cpu_ones + cpu_ones
    ort_twos = ort_ones + ort_ones
    assert torch.allclose(cpu_twos, ort_twos.cpu())
  
  def test_type_promotion_add(self):
    device = self.get_device()
    x = torch.ones(2, 5, dtype = torch.int64)
    y = torch.ones(2, 5, dtype = torch.float32)
    ort_x = x.to(device)
    ort_y = y.to(device)
    ort_z = ort_x + ort_y
    assert ort_z.dtype == torch.float32
    assert torch.allclose(ort_z.cpu(), (x + y))

  def test_add_alpha(self):
    device = self.get_device()
    cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    ort_ones = cpu_ones.to(device)
    assert torch.allclose(
      torch.add(cpu_ones, cpu_ones, alpha=2.5),
      torch.add(ort_ones, ort_ones, alpha=2.5).cpu())
  
  def test_mul_bool(self):
    device = self.get_device()
    cpu_ones = torch.ones(3, 3, dtype=bool)
    ort_ones = cpu_ones.to(device)
    assert torch.allclose(
      torch.mul(cpu_ones, cpu_ones),
      torch.mul(ort_ones, ort_ones).cpu())

  # TODO: Add BFloat16 test coverage
  def test_add_(self):
    device = self.get_device()
    cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    ort_ones = cpu_ones.to(device)
    cpu_twos = cpu_ones
    cpu_twos += cpu_ones
    ort_twos = ort_ones
    ort_twos += ort_ones
    assert torch.allclose(cpu_twos, ort_twos.cpu())

  def test_sin_(self):
    device = self.get_device()
    cpu_sin_pi_ = torch.Tensor([np.pi])
    torch.sin_(cpu_sin_pi_)
    ort_sin_pi_ = torch.Tensor([np.pi]).to(device)
    torch.sin_(ort_sin_pi_)
    cpu_sin_pi = torch.sin(torch.Tensor([np.pi]))
    ort_sin_pi = torch.sin(torch.Tensor([np.pi]).to(device))
    assert torch.allclose(cpu_sin_pi, ort_sin_pi.cpu())
    assert torch.allclose(cpu_sin_pi_, ort_sin_pi_.cpu())
    assert torch.allclose(ort_sin_pi.cpu(), ort_sin_pi_.cpu())

  def test_sin(self):
    device = self.get_device()
    cpu_sin_pi = torch.sin(torch.Tensor([np.pi]))
    ort_sin_pi = torch.sin(torch.Tensor([np.pi]).to(device))
    assert torch.allclose(cpu_sin_pi, ort_sin_pi.cpu())
  
  def test_zero_like(self):
    device = self.get_device()
    ones = torch.ones((10, 10), dtype=torch.float32)
    cpu_zeros = torch.zeros_like(ones)
    ort_zeros = torch.zeros_like(ones.to(device))
    assert torch.allclose(cpu_zeros, ort_zeros.cpu())

  def test_gemm(self):
    device = self.get_device()
    cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    ort_ones = cpu_ones.to(device)
    cpu_ans = cpu_ones * 4
    ort_ans = torch_ort.custom_ops.gemm(ort_ones, ort_ones, ort_ones, 1.0, 1.0, 0, 0)
    assert torch.allclose(cpu_ans, ort_ans.cpu())
  
  def test_max(self):
    cpu_tensor = torch.rand(10, 10)
    ort_tensor = cpu_tensor.to('ort')
    y = ort_tensor.max()
    x = cpu_tensor.max()
    assert torch.allclose(x, y.cpu())
  
  def test_min(self):
    cpu_tensor = torch.rand(10, 10)
    ort_tensor = cpu_tensor.to('ort')
    y = ort_tensor.min()
    x = cpu_tensor.min()
    assert torch.allclose(x, y.cpu())

if __name__ == '__main__':
  unittest.main()