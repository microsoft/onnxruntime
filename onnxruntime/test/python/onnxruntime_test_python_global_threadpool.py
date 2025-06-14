# pylint: disable=C0115,W0212,C0103,C0114
import unittest

import numpy as np
from helper import get_name

import onnxruntime as onnxrt


class TestGlobalThreadPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        onnxrt.set_global_thread_pool_sizes(2, 2)

    def test_global_threadpool(self):
        session_opts = onnxrt.SessionOptions()
        session_opts.execution_mode = onnxrt.ExecutionMode.ORT_PARALLEL
        session_opts.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_DISABLE_ALL
        session_opts.use_per_session_threads = False
        session = onnxrt.InferenceSession(
            get_name("mnist.onnx"), session_opts, providers=onnxrt.get_available_providers()
        )
        input = np.ones([1, 1, 28, 28], np.float32)
        session.run(None, {"Input3": input})

    def test_raise_error_if_use_per_session_threads(self):
        session_opts = onnxrt.SessionOptions()
        session_opts.execution_mode = onnxrt.ExecutionMode.ORT_PARALLEL
        session_opts.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_DISABLE_ALL
        session_opts.use_per_session_threads = True
        with self.assertRaises(RuntimeError):
            onnxrt.InferenceSession(get_name("mnist.onnx"), session_opts, providers=onnxrt.get_available_providers())


if __name__ == "__main__":
    unittest.main()
