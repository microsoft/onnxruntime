# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import numpy as np
import gc

import onnxruntime as onnxrt
import threading
import sys
from helper import get_name
from onnxruntime.capi.onnxruntime_pybind11_state import Fail
import time

class TestInferenceSessionWithCudaGraph(unittest.TestCase):
  def testOrtValueUpdateInPlace(self):
      x0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
      ortvalue_cpu = onnxrt.OrtValue.ortvalue_from_numpy(x0)
      np.testing.assert_allclose(x0, ortvalue_cpu.numpy())
      
      x1 = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
      ortvalue_cpu.update_inplace(x1)
      np.testing.assert_allclose(x1, ortvalue_cpu.numpy())
      
      if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
          ortvalue_gpu = onnxrt.OrtValue.ortvalue_from_numpy(x0, 'cuda', 0)
          np.testing.assert_allclose(x0, ortvalue_gpu.numpy())
          
          ortvalue_gpu.update_inplace(x1)
          np.testing.assert_allclose(x1, ortvalue_gpu.numpy())
      
  def testRunModelWithCudaGraph(self):
      if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
          warmup_runs = 10
          providers = [('CUDAExecutionProvider', {'enable_cuda_graph': True, 'cuda_graph_warmup_runs': warmup_runs})]
          INPUT_SIZE = 1280
          x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]*INPUT_SIZE, dtype=np.float32)
          y = np.array([[0.0], [0.0], [0.0]]*INPUT_SIZE, dtype=np.float32)
          x_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
          y_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(y, 'cuda', 0)

          session = onnxrt.InferenceSession(get_name("matmul_2.onnx"), providers=providers)
          io_binding = session.io_binding()

          # Bind the input and output
          io_binding.bind_input('X', 'cuda', 0, np.float32, [INPUT_SIZE*3, 2], x_ortvalue.data_ptr())
          io_binding.bind_output('Y', 'cuda', 0, np.float32, [INPUT_SIZE*3, 1], y_ortvalue.data_ptr())

          # RUN - 0
          # Warm-up Run() - (CUDA Graph capture happens after the warmup runs finish)
          for _ in range(warmup_runs):
            session.run_with_iobinding(io_binding)
          expected_y = np.array([[5.0], [11.0], [17.0]]*INPUT_SIZE, dtype=np.float32)
          np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)
          
          # RUN - 1
          # Capture the CUDA Graph
          session.run_with_iobinding(io_binding)
          np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

          # RUN - 2
          # Graph Replay Run 1 (CUDA Graph replay happens from this Run onwards)
          session.run_with_iobinding(io_binding)          
          np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)


          # RUN - 3
          # Graph Replay Run 2 (CUDA Graph replay)
          # Update input
          x_ortvalue.update_inplace(np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]*INPUT_SIZE, dtype=np.float32))
          session.run_with_iobinding(io_binding)
          np.testing.assert_allclose(np.array([[50.0], [110.0], [170.0]]*INPUT_SIZE, dtype=np.float32), y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

if __name__ == '__main__':
    unittest.main()
