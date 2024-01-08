# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import sys
import unittest

import numpy as np
import onnx

from onnxruntime import InferenceSession, SessionOptions


class TestOnnxToolsGraph(unittest.TestCase):
    def test_basic_all(self):
        if sys.platform.startswith("win"):
            shared_library = "custom_op_local_function.dll"
        elif sys.platform.startswith("darwin"):
            shared_library = "libcustom_op_local_function.dylib"
        else:
            shared_library = "./libcustom_op_local_function.so"
        if not os.path.exists(shared_library):
            raise FileNotFoundError(f"Unable to find '{shared_library}'")

        filename = "custom_ops_type_inference_fails_0.onnx"

        with open(os.path.join(os.path.dirname(__file__), filename), "rb") as f:
            onxo = onnx.load(f)
        d = onxo.opset_import.add()
        d.domain = "ai.onnx.ml"
        d.version = 2

        sess_opts = SessionOptions()
        sess_opts.register_custom_ops_library(shared_library)

        sess = InferenceSession(
            onxo.SerializeToString(),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )
        x = np.arange(2**2).reshape((2,) * 2).astype(np.float32)
        t = np.arange(8).reshape((2, 4)).astype(np.float32)
        got = sess.run(None, dict(X=x))[0]
        np.testing.assert_allclose(t, got, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
