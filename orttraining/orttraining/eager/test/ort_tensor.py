# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import torch
import onnxruntime_pybind11_state as torch_ort

class OrtTensorTests(unittest.TestCase):
  def test_is_ort_via_alloc(self):
    cpu_ones = torch.zeros(10, 10)
    assert not cpu_ones.is_ort
    ort_ones = torch.zeros(10, 10, device='ort')
    assert ort_ones.is_ort
    assert torch.allclose(cpu_ones, ort_ones.cpu())

  def test_is_ort_via_to(self):
    cpu_ones = torch.ones(10, 10)
    assert not cpu_ones.is_ort
    ort_ones = cpu_ones.to('ort')
    assert ort_ones.is_ort
    assert torch.allclose(cpu_ones, ort_ones.cpu())

  def test_reshape(self):
    cpu_ones = torch.ones(10, 10)
    ort_ones = cpu_ones.to('ort')
    y = ort_ones.reshape(-1)
    assert len(y.size()) == 1
    assert y.size()[0] == 100

  def test_view(self):
    cpu_ones = torch.ones(2048)
    ort_ones = cpu_ones.to('ort')
    y = ort_ones.view(4, 512)
    assert y.size() == (4, 512)

  def test_view_neg1(self):
    cpu_ones = torch.ones(784, 256)
    ort_ones = cpu_ones.to('ort')
    y = ort_ones.view(-1)
    assert y.size()[0] == 200704

  def test_stride(self):
    cpu_ones = torch.ones(3, 3)
    ort_ones = cpu_ones.to('ort')
    y = torch.as_strided(ort_ones, (2, 2), (1, 2))
    assert y.size() == (2, 2)
    assert y.is_contiguous() == False
    contiguous_y = y.contiguous()
    w = torch.ones((2,3))
    ort_w = w.to('ort')
    z = torch.zeros((2, 3))
    ort_z = z.to('ort')
    ort_z = torch.addmm(ort_z, contiguous_y, ort_w)
    cpu_z = torch.addmm(z, torch.ones(2, 2), w)
    assert torch.allclose(ort_z.cpu(), cpu_z)

  def test_slice(self):
    cpu_ones = torch.ones((128, 256), dtype=torch.bfloat16)
    ort_ones = cpu_ones.to('ort')
    y_cpu = cpu_ones[0:128, :128]
    y = ort_ones[0:128, :128]
    assert y.is_contiguous() == False
    assert y.size() == (128, 128)
    assert torch.allclose(y.cpu(), y_cpu)

if __name__ == '__main__':
  unittest.main()
