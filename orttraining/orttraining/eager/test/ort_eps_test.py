# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import torch
import onnxruntime_pybind11_state as torch_ort
import os

class OrtEPTests(unittest.TestCase):
  def get_test_execution_provider_path(self):
      return os.path.join('.', 'libtest_execution_provider.so')

  def test_import_custom_eps(self):
    torch_ort.set_device(0, 'CPUExecutionProvider', {})

    torch_ort._register_provider_lib('TestExecutionProvider', self.get_test_execution_provider_path(), {})
    torch_ort.set_device(1, 'TestExecutionProvider', {'device_id':'0', 'some_config':'val'})
    ort_device = torch_ort.device(1)

if __name__ == '__main__':
  unittest.main()