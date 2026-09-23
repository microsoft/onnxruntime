import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from drift_investigation.materialize_static_uint16 import materialize


class MaterializeStaticUint16Test(unittest.TestCase):
    def test_materializes_only_static_uint16_and_preserves_cpu_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.onnx"
            output = Path(folder) / "float_constants.onnx"
            graph = helper.make_graph(
                [
                    helper.make_node("DequantizeLinear", ["q", "scale", "zp"], ["float_c"]),
                    helper.make_node("Mul", ["x", "float_c"], ["y"]),
                ],
                "static_dq",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
                initializer=[
                    numpy_helper.from_array(np.array([0, 200], dtype=np.uint16), "q"),
                    numpy_helper.from_array(np.array(0.25, dtype=np.float32), "scale"),
                    numpy_helper.from_array(np.array(100, dtype=np.uint16), "zp"),
                ],
            )
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)]), source)
            self.assertEqual(materialize(source, output, 1), 1)
            self.assertFalse(output.with_name(output.name + ".data").exists())
            proto = onnx.load(output)
            self.assertEqual([node.op_type for node in proto.graph.node], ["Mul"])
            inputs = {"x": np.array([0.5, -2.0], dtype=np.float32)}
            original = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
            variant = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
            np.testing.assert_array_equal(original, variant)
            with self.assertRaises(FileExistsError):
                materialize(source, output, 1)

    def test_rejects_quantized_runtime_activation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.onnx"
            graph = helper.make_graph(
                [
                    helper.make_node("QuantizeLinear", ["x", "scale", "zp"], ["q"]),
                    helper.make_node("DequantizeLinear", ["q", "scale", "zp"], ["y"]),
                ],
                "activation_qdq",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
                initializer=[
                    numpy_helper.from_array(np.array(0.25, dtype=np.float32), "scale"),
                    numpy_helper.from_array(np.array(0, dtype=np.uint16), "zp"),
                ],
            )
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)]), source)
            with self.assertRaisesRegex(ValueError, "activation QuantizeLinear"):
                materialize(source, Path(folder) / "output.onnx")


if __name__ == "__main__":
    unittest.main()
