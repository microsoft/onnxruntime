# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import torch
import onnxruntime_pybind11_state as torch_ort
import os
import sys

def is_windows():
    return sys.platform.startswith("win")

class OrtEPTests(unittest.TestCase):
  def get_test_execution_provider_path(self):
      if is_windows():
        return os.path.join('.', 'test_execution_provider.dll')
      else:
        return os.path.join('.', 'libtest_execution_provider.so')

  def test_import_custom_eps(self):
    torch_ort.set_device(0, 'CPUExecutionProvider', {})

    torch_ort._register_provider_lib('TestExecutionProvider', self.get_test_execution_provider_path(), {'some_config':'val'})
    torch_ort.set_device(1, 'TestExecutionProvider', {'device_id':'0', 'some_config':'val'})
    ort_device = torch_ort.device(1)

if __name__ == '__main__':
  unittest.main()