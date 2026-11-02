# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from compare_decoder import compare
from onnx import TensorProto, helper, numpy_helper
from optimize_decoder import optimize, prune_graph

import onnxruntime as ort


def decoder(rank=0, width=8, different_input=False, expose_lora=False):
    nodes = [
        helper.make_node(
            "MatMulNBits", ["x", "w", "scales"], ["base"], domain="com.microsoft", K=32, N=8, bits=4, block_size=32
        ),
        helper.make_node("Cast", ["la"], ["la_cast"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["other" if different_input else "x", "la_cast"], ["a"]),
        helper.make_node("MatMul", ["a", "lb"], ["b"]),
        helper.make_node("Add", ["base", "b"], ["sum"]),
        helper.make_node("Identity", ["sum"], ["y"]),
    ]
    rng = np.random.default_rng(42)
    tensors = [
        numpy_helper.from_array(rng.integers(0, 256, (8, 1, 16), dtype=np.uint8), "w"),
        numpy_helper.from_array(np.ones(8, np.float32) * 0.1, "scales"),
        numpy_helper.from_array(np.zeros((32, rank), np.float16), "la"),
        numpy_helper.from_array(np.zeros((rank, width), np.float32), "lb"),
    ]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 32])]
    if different_input:
        inputs.append(helper.make_tensor_value_info("other", TensorProto.FLOAT, ["batch", 32]))
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 8])]
    if expose_lora:
        outputs.append(helper.make_tensor_value_info("b", TensorProto.FLOAT, ["batch", width]))
    # Deliberately non-topological: exported Aion is also not dependency-ordered.
    graph = helper.make_graph(list(reversed(nodes)), "decoder", inputs, outputs, tensors)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=10
    )


class OptimizeDecoderTest(unittest.TestCase):
    def test_empty_branch_pruned_and_numerically_equal(self):
        model = decoder()
        original = copy.deepcopy(model)
        # Order the reference without applying the transformation.
        prune_graph(original.graph)
        result = optimize(model)
        self.assertEqual(result["empty_lora_branches_removed"], 1)
        self.assertEqual([node.op_type for node in model.graph.node], ["MatMulNBits", "Identity"])
        self.assertEqual({value.name for value in model.graph.initializer}, {"w", "scales"})
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sessions = [
            ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"])
            for m in (original, model)
        ]
        x = np.random.default_rng(0).normal(size=(3, 32)).astype(np.float32)
        np.testing.assert_array_equal(sessions[0].run(None, {"x": x})[0], sessions[1].run(None, {"x": x})[0])

    def test_nonempty_mismatched_and_other_input_not_removed(self):
        for kwargs in ({"rank": 1}, {"width": 7}, {"different_input": True}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(optimize(decoder(**kwargs))["empty_lora_branches_removed"], 0)

    def test_visible_lora_output_stays_live(self):
        model = decoder(expose_lora=True)
        self.assertEqual(optimize(model)["empty_lora_branches_removed"], 1)
        self.assertIn("b", {value for node in model.graph.node for value in node.output})
        self.assertNotIn("Add", [node.op_type for node in model.graph.node])
        onnx.checker.check_model(model)

    def test_add_output_name_preserved(self):
        model = decoder()
        model.graph.output[0].name = "sum"
        optimize(model)
        self.assertEqual(model.graph.node[-1].op_type, "Identity")
        self.assertEqual(list(model.graph.node[-1].output), ["sum"])
        onnx.checker.check_model(model)

    def test_overridable_weight_not_assumed_constant(self):
        model = decoder()
        model.graph.input.append(helper.make_tensor_value_info("la", TensorProto.FLOAT16, [32, "rank"]))
        self.assertEqual(optimize(model)["empty_lora_branches_removed"], 0)
        result = optimize(model, freeze_initializers=True)
        self.assertEqual(result["initializer_inputs_frozen"], 1)
        self.assertEqual(result["empty_lora_branches_removed"], 1)

    def test_specialization_only_changes_named_head_dimension(self):
        model = decoder()
        model.graph.input.append(
            helper.make_tensor_value_info("past", TensorProto.FLOAT, ["batch", 8, "past_length", "kv_cache_head_size"])
        )
        result = optimize(model, 128)
        dims = model.graph.input[-1].type.tensor_type.shape.dim
        self.assertEqual(result["kv_head_dimensions_specialized"], 1)
        self.assertEqual(dims[-1].dim_value, 128)
        self.assertEqual(dims[-2].dim_param, "past_length")

    def test_idempotent(self):
        model = decoder()
        optimize(model)
        state = model.SerializeToString()
        self.assertEqual(optimize(model)["empty_lora_branches_removed"], 0)
        self.assertEqual(model.SerializeToString(), state)


class CompareDecoderTest(unittest.TestCase):
    def test_strict_comparison_includes_zero_sign_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.npz", Path(directory) / "b.npz"
            np.savez(a, logits=np.array([[0.0, 1.0]], dtype=np.float16))
            np.savez(b, logits=np.array([[0.0, 1.0]], dtype=np.float16))
            self.assertTrue(compare(a, b)["byte_exact"])
            np.savez(b, logits=np.array([[-0.0, 1.0]], dtype=np.float16))
            result = compare(a, b)
            self.assertFalse(result["byte_exact"])
            self.assertEqual(result["outputs"][0]["max_abs"], 0)
            np.savez(b, logits=np.array([[np.nan, 1.0]], dtype=np.float16))
            result = compare(a, b)
            self.assertFalse(result["outputs"][0]["finite"])
            self.assertIsNone(result["outputs"][0]["max_abs"])

    def test_mismatched_names_shapes_and_types_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.npz", Path(directory) / "b.npz"
            np.savez(a, logits=np.array([[1.0]], dtype=np.float16))
            for arrays in (
                {"other": np.ones((1, 1), np.float16)},
                {"logits": np.ones((2, 1), np.float16)},
                {"logits": np.ones((1, 1), np.float32)},
            ):
                np.savez(b, **arrays)
                with self.assertRaises(ValueError):
                    compare(a, b)


if __name__ == "__main__":
    unittest.main()
