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
          
  def testRunModelWithFeedsFetches(self):
      def invoke(device):
          providers = ["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"]
          sess = onnxrt.InferenceSession(get_name("matmul_1.onnx"), providers=providers)
          x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
          y = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
          x_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(x, device, 0)
          y_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(y, device, 0)
          
          feeds = {"X": x_ortvalue}
          fetches = {"Y": y_ortvalue}
          
          sess.run_with_feeds_fetches_ort_values(feeds, fetches)
          
          expected_y = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
          np.testing.assert_allclose(expected_y, fetches["Y"].numpy(), rtol=1e-05, atol=1e-05)
      
      invoke("cpu")
      if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
          invoke("cuda")  
      
  def testRunModelWithCudaGraph(self):
      if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
          providers = ["CUDAExecutionProvider"]
          sess = onnxrt.InferenceSession(get_name("matmul_2.onnx"), providers=providers)
          INPUT_SIZE = 1280
          x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]*INPUT_SIZE, dtype=np.float32)
          y = np.array([[0.0], [0.0], [0.0]]*INPUT_SIZE, dtype=np.float32)
          x_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
          y_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(y, 'cuda', 0)
          
          feeds = {"X": x_ortvalue}
          fetches = {"Y": y_ortvalue}
          
          for _ in range(1):
            sess.run_with_feeds_fetches_ort_values(feeds, fetches)
          
          expected_y = np.array([[5.0], [11.0], [17.0]]*INPUT_SIZE, dtype=np.float32)
          np.testing.assert_allclose(expected_y, fetches["Y"].numpy(), rtol=1e-05, atol=1e-05)
          
          sess.turn_on_capture()      
          sess.run_with_feeds_fetches_ort_values(feeds, fetches)
          sess.turn_off_capture()
          sess.replay()
          np.testing.assert_allclose(expected_y, fetches["Y"].numpy(), rtol=1e-05, atol=1e-05)
          
          x = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]*INPUT_SIZE, dtype=np.float32)
          expected_y = np.array([[50.0], [110.0], [170.0]]*INPUT_SIZE, dtype=np.float32)
          x_ortvalue.update_inplace(x)
          sess.replay()
          np.testing.assert_allclose(expected_y, fetches["Y"].numpy(), rtol=1e-05, atol=1e-05)

if __name__ == '__main__':
    unittest.main()
