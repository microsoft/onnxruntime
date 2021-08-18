# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This test must run as a separate process since we assert ORT things
# are unavailable before we import torch_ort, then test that they do
# in fact become available after importing it. The act of importing
# torch_ort makes it available implicitly to any tests that may run
# after the import, hence this test is isolated from the others.

import unittest
import torch

class OrtInitTests(unittest.TestCase):
  def test_ort_init(self):
    config_match = 'ORT is enabled'

    def ort_alloc():
      torch.zeros(5, 5, device='ort')

    self.assertNotIn(config_match, torch._C._show_config())
    with self.assertRaises(BaseException):
      ort_alloc()

    import onnxruntime_pybind11_state as torch_ort
    ort_alloc()
    self.assertIn(config_match, torch._C._show_config())

if __name__ == '__main__':
  unittest.main()
