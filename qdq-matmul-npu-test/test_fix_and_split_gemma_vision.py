"""Regression tests for the Gemma vision padding-mask QDQ repair."""

import unittest

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx import helper, numpy_helper

from fix_and_split_gemma_vision import fix_mask_qdq


def _mask_fixture() -> ir.Model:
    prefix = "v_vision_encoder.encoder."
    constant = f"{prefix}CastLike_29"
    mask = f"{prefix}Where_31"
    unsqueeze = f"{prefix}Unsqueeze_32"
    nodes = [
        helper.make_node(
            "DequantizeLinear",
            [f"{constant}_quantized", f"{constant}_scale", f"{constant}_zero_point"],
            [f"{constant}_DequantizeLinear_Output"],
            name=f"{constant}_DequantizeLinear",
        ),
        helper.make_node(
            "Where",
            ["padded", f"{constant}_DequantizeLinear_Output", "zero"],
            ["mask_value"],
            name="vision_encoder/encoder/Where_node_31",
        ),
        helper.make_node(
            "QuantizeLinear", ["mask_value", f"{mask}_scale", f"{mask}_zero_point"], ["mask_q"],
            name=f"{mask}_QuantizeLinear",
        ),
        helper.make_node(
            "DequantizeLinear", ["mask_q", f"{mask}_scale", f"{mask}_zero_point"], ["mask_dq"],
            name=f"{mask}_DequantizeLinear",
        ),
        helper.make_node("Unsqueeze", ["mask_dq", "axes"], ["expanded"], name="Unsqueeze_32"),
        helper.make_node(
            "QuantizeLinear", ["expanded", f"{mask}_scale", f"{mask}_zero_point"], ["expanded_q"],
            name=f"{unsqueeze}_QuantizeLinear",
        ),
        helper.make_node(
            "DequantizeLinear", ["expanded_q", f"{mask}_scale", f"{mask}_zero_point"], ["output"],
            name=f"{unsqueeze}_DequantizeLinear",
        ),
    ]
    initializers = {
        f"{constant}_quantized": np.asarray(0, dtype=np.uint16),
        f"{constant}_scale": np.asarray(1e9 / 65535, dtype=np.float32),
        f"{constant}_zero_point": np.asarray(65535, dtype=np.uint16),
        f"{mask}_scale": np.asarray(1.0, dtype=np.float32),
        f"{mask}_zero_point": np.asarray(0, dtype=np.uint16),
        "zero": np.asarray(0, dtype=np.float32),
        "axes": np.asarray([0], dtype=np.int64),
    }
    graph = helper.make_graph(
        nodes, "mask_qdq",
        [helper.make_tensor_value_info("padded", onnx.TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
        initializer=[numpy_helper.from_array(value, name) for name, value in initializers.items()],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 21)], ir_version=10)
    return ir.serde.deserialize_model(proto)


class FixMaskQdqTest(unittest.TestCase):
    def test_corrected_mask_survives_both_qdq_pairs(self):
        for mask_value in (-100, -1000, -65535):
            with self.subTest(mask_value=mask_value):
                model = _mask_fixture()
                original = ort.InferenceSession(
                    ir.serde.serialize_model(model).SerializeToString(), providers=["CPUExecutionProvider"]
                )
                self.assertEqual(original.run(None, {"padded": np.asarray(True)})[0].item(), 0.0)

                fix_mask_qdq(model, mask_value)
                corrected = ort.InferenceSession(
                    ir.serde.serialize_model(model).SerializeToString(), providers=["CPUExecutionProvider"]
                )
                for padded, expected in [(True, float(mask_value)), (False, 0.0)]:
                    result = corrected.run(None, {"padded": np.asarray(padded)})[0]
                    self.assertEqual(result.item(), expected)

    def test_rejects_unexpected_graph_without_mutating_it(self):
        model = _mask_fixture()
        model.graph.initializers["v_vision_encoder.encoder.Where_31_zero_point"].const_value = ir.tensor(
            np.asarray(1, dtype=np.uint16)
        )
        with self.assertRaisesRegex(ValueError, "expected original value"):
            fix_mask_qdq(model)
        self.assertEqual(
            model.graph.initializers["v_vision_encoder.encoder.CastLike_29_scale"].const_value.numpy().item(),
            np.float32(1e9 / 65535).item(),
        )

    def test_rejects_out_of_range_mask(self):
        for value in (0, -65536):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "negative integer"):
                fix_mask_qdq(_mask_fixture(), value)


if __name__ == "__main__":
    unittest.main()
