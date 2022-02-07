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
    ort_twos = torch_ort.onnx.Add(ort_ones, ort_ones)
    assert torch.allclose(cpu_twos, ort_twos.cpu())
  
  def test_reducesum(self):
    device = self.get_device()
    torch_x = torch.ones(3, 3)
    ort_x = torch_x.to('ort')
    axis = torch.Tensor([0]).to(torch.int64).to('ort')
    y = torch_ort.onnx.ReduceSum(ort_x, axis, 0, 0)
    torch_sum = torch.sum(torch_x, 0)
    assert torch.allclose(y.cpu(), torch_sum)

if __name__ == '__main__':
  unittest.main()