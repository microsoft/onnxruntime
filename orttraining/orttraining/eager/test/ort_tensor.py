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

if __name__ == '__main__':
  unittest.main()