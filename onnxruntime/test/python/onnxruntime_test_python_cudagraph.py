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
          providers = [('CUDAExecutionProvider', {'enable_cuda_graph': True})]
          INPUT_SIZE = 1280
          x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]*INPUT_SIZE, dtype=np.float32)
          y = np.array([[0.0], [0.0], [0.0]]*INPUT_SIZE, dtype=np.float32)
          x_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
          y_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(y, 'cuda', 0)

          session = onnxrt.InferenceSession(get_name("matmul_2.onnx"), providers=providers)
          io_binding = session.io_binding()

          # Bind the input and output
          io_binding.bind_ortvalue_input('X', x_ortvalue)
          io_binding.bind_ortvalue_output('Y', y_ortvalue)

          # One regular run for the necessary memory allocation before cuda graph capture
          session.run_with_iobinding(io_binding)
          expected_y = np.array([[5.0], [11.0], [17.0]]*INPUT_SIZE, dtype=np.float32)
          np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)
          
          # This run captures the CUDA graph
          session.run_with_iobinding(io_binding)
          np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

          # After capturing, CUDA graph replay happens from this Run onwards
          session.run_with_iobinding(io_binding)
          np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

          # Update input and then replay CUDA graph
          x_ortvalue.update_inplace(np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]*INPUT_SIZE, dtype=np.float32))
          session.run_with_iobinding(io_binding)
          np.testing.assert_allclose(np.array([[50.0], [110.0], [170.0]]*INPUT_SIZE, dtype=np.float32), y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

  def testRunModelWithCudaGraphWithMultipleThreads(self):
      if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
          providers = [('CUDAExecutionProvider', {'enable_cuda_graph': True})]
          INPUT_SIZE = 1
          x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]*INPUT_SIZE, dtype=np.float32)
          y = np.array([[0.0], [0.0], [0.0]]*INPUT_SIZE, dtype=np.float32)

          so = onnxrt.SessionOptions()
          so.log_verbosity_level = 0
          so.logid = "MultiThreadsTest"
          session = onnxrt.InferenceSession(get_name("matmul_2.onnx"), sess_options=so, providers=providers)

          num_threads = 4
          io_bindings = [session.io_binding() for _ in range(num_threads)]
          x_ortvalues = [onnxrt.OrtValue.ortvalue_from_numpy(x*i, 'cuda', 0) for i in range(num_threads)]
          y_ortvalues = [onnxrt.OrtValue.ortvalue_from_numpy(y, 'cuda', 0) for _ in range(num_threads)]
          ros = [onnxrt.RunOptions() for _ in range(num_threads)]
          for i, ro in enumerate(ros):
              ro.logid = "thread" + str(i)
          for i, io_binding in enumerate(io_bindings):
              io_binding.bind_ortvalue_input('X', x_ortvalues[i])
              io_binding.bind_ortvalue_output('Y', y_ortvalues[i])

          def sess_run(io_binding, ro):
              for _ in range(10000):
                  session.run_with_iobinding(io_binding, ro)

          def run_multi_threads():
            run_threads = []

            for i in range(num_threads):
                t = threading.Thread(target=sess_run, args=(io_bindings[i], ros[i]))
                run_threads.append(t)

            for t in run_threads:
                t.start()

            for t in run_threads:
                t.join()

          run_multi_threads()

          for i, output in enumerate(y_ortvalues):
              np.testing.assert_allclose(
                  np.array([[5.0*i], [11.0*i], [17.0*i]]*INPUT_SIZE, dtype=np.float32),
                  output.numpy(),
                  rtol=1e-05,
                  atol=1e-05)

if __name__ == '__main__':
    unittest.main()
