# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0116,W0212,R1720,C0103,C0114

import platform
import unittest
import numpy as np
from numpy.testing import assert_allclose
import onnx
from onnx import TensorProto
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model
import onnxruntime


# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:
    os.add_dll_directory(os.getcwd())

available_providers = [provider for provider in onnxruntime.get_available_providers()]


class TestInferenceSession(unittest.TestCase):
    def model_cast_cast(self, to):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("Cast", ["X"], ["T"], to=to)
        node2 = make_node("Cast", ["T"], ["Y"], to=TensorProto.FLOAT)
        graph = make_graph([node1, node2], "lr", [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(not hasattr(TensorProto, "FLOATE4M3"), reason="needs onnx>=1.4.0")
    def test_model_cast_cast(self):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        expected = {
            TensorProto.FLOATE4M3: np.array(
                [
                    0.000000e00,
                    -1.000000e00,
                    1.015625e-01,
                    1.000000e00,
                    2.000000e00,
                    1.000000e01,
                    1.040000e02,
                    4.480000e02,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOATE5M2: np.array(
                [0.000e00, -1.000e00, 9.375e-02, 1.000e00, 2.000e00, 1.000e01, 9.600e01, 1.024e03], dtype=np.float32
            ),
        }
        x = np.array([0, 1 - 2, 1e-1, 1, 2, 10, 100, 1000], dtype=np.float32)

        for to, expect in expected.items():
            onnx_model = self.model_cast_cast(to)
            for prov in ["CPUExecutionProvider", "CUDAExecutionProvider"]:
                if prov not in available_providers:
                    continue
                with self.subTest(provider=prov, to=to):
                    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=[prov])
                    y = sess.run(None, {"X": x})[0]
                    assert_allclose(y, expect)


if __name__ == "__main__":
    unittest.main()
