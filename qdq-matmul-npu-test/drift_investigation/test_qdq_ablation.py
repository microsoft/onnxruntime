"""Provider-independent QDQ ablation and input/report regression tests."""

import re
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx import helper, numpy_helper

from drift_investigation.qdq_ablation import (
    activation_pairs, attention_core_pairs, bypass_pairs, select_pairs, write_variant,
)
from run_acc import (
    compare_outputs, context_node_count, device_identity, load_inputs, measure_float_output,
    run_session, validate_compiled_source,
)


def make_model(qdq_domain: str = "") -> onnx.ModelProto:
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zp"], ["a_q"], name="group_a_Q", domain=qdq_domain),
        helper.make_node("DequantizeLinear", ["a_q", "scale", "zp"], ["a_dq"], name="group_a_DQ", domain=qdq_domain),
        helper.make_node("Add", ["a_dq", "bias"], ["intermediate"], name="add_a"),
        helper.make_node(
            "QuantizeLinear", ["intermediate", "scale", "zp"], ["b_q"],
            name="group_b_Q", domain=qdq_domain,
        ),
        helper.make_node("DequantizeLinear", ["b_q", "scale", "zp"], ["b_dq"], name="group_b_DQ", domain=qdq_domain),
        helper.make_node(
            "DequantizeLinear", ["weight", "weight_scale", "weight_zp"], ["weight_float"],
            name="weight_DQ", domain=qdq_domain,
        ),
        helper.make_node("Add", ["b_dq", "weight_float"], ["y"], name="add_b"),
    ]
    values = {
        "scale": np.asarray(0.5, dtype=np.float32),
        "zp": np.asarray(128, dtype=np.uint8),
        "bias": np.asarray(0.25, dtype=np.float32),
        "weight": np.asarray(3, dtype=np.uint8),
        "weight_scale": np.asarray(0.5, dtype=np.float32),
        "weight_zp": np.asarray(0, dtype=np.uint8),
    }
    graph = helper.make_graph(
        nodes, "ablation",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])],
        initializer=[numpy_helper.from_array(value, name) for name, value in values.items()],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 21)] + (
            [helper.make_operatorsetid("com.microsoft", 1)] if qdq_domain else []
        ),
        ir_version=10,
    )


def make_attention_model() -> onnx.ModelProto:
    nodes = [helper.make_node("MatMul", ["query", "keys"], ["logits"], name="qk")]
    for input_name, output_name, prefix in (
        ("logits", "score_dq", "score"),
        ("masked", "masked_dq", "mask_add"),
        ("probabilities", "prob_dq", "softmax"),
    ):
        nodes.extend((
            helper.make_node("QuantizeLinear", [input_name, "scale", "zp"], [f"{prefix}_q"], name=f"{prefix}_Q"),
            helper.make_node(
                "DequantizeLinear", [f"{prefix}_q", "scale", "zp"], [output_name], name=f"{prefix}_DQ"
            ),
        ))
    nodes.extend((
        helper.make_node("Add", ["score_dq", "mask"], ["masked"], name="attention_mask"),
        helper.make_node("Softmax", ["masked_dq"], ["probabilities"], name="attention_softmax", axis=-1),
        helper.make_node("MatMul", ["prob_dq", "values"], ["attention_result"], name="pv"),
        helper.make_node("Identity", ["attention_result"], ["output"], name="encoder_output"),
    ))
    graph = helper.make_graph(
        nodes, "attention",
        [
            helper.make_tensor_value_info("query", onnx.TensorProto.FLOAT, [1, 2, 2]),
            helper.make_tensor_value_info("keys", onnx.TensorProto.FLOAT, [1, 2, 2]),
            helper.make_tensor_value_info("values", onnx.TensorProto.FLOAT, [1, 2, 2]),
        ],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2, 2])],
        initializer=[
            numpy_helper.from_array(np.asarray(1 / 255, dtype=np.float32), "scale"),
            numpy_helper.from_array(np.asarray(0, dtype=np.uint8), "zp"),
            numpy_helper.from_array(np.asarray(0, dtype=np.float32), "mask"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 21)], ir_version=10)


class QdqAblationTest(unittest.TestCase):
    def test_device_identity_captures_selected_hardware(self):
        hardware = SimpleNamespace(vendor="Example", vendor_id=123, device_id=456, type="NPU")
        device = SimpleNamespace(ep_vendor="Provider", device=hardware)
        self.assertEqual(device_identity(device), {
            "ep_vendor": "Provider", "hardware_vendor": "Example",
            "vendor_id": 123, "device_id": 456, "device_type": "NPU",
        })

    def setUp(self):
        self.inputs = {"x": np.asarray([[-0.4, 0.25, 1.1, 4.0]], dtype=np.float32)}

    def test_bypass_one_group_and_keep_other_group(self):
        original = make_model()
        model = ir.serde.deserialize_model(original)
        pairs = activation_pairs(model.graph)
        self.assertEqual([pair.name for pair in pairs], ["group_a_Q", "group_b_Q"])
        selected = select_pairs(pairs, re.compile("group_a"), None, [re.compile("group_b")])
        self.assertEqual([pair.name for pair in selected], ["group_a_Q"])
        self.assertEqual(
            [pair.name for pair in select_pairs(pairs, None, re.compile("group_b"), [])],
            ["group_a_Q"],
        )
        self.assertEqual(
            [pair.name for pair in select_pairs(pairs, None, None, [], {"group_b_Q"})],
            ["group_b_Q"],
        )
        with self.assertRaisesRegex(ValueError, "unknown activation QDQ pairs"):
            select_pairs(pairs, None, None, [], {"missing_Q"})
        bypass_pairs(model, selected)
        variant = ir.serde.serialize_model(model)
        self.assertEqual(sum(node.op_type == "DequantizeLinear" for node in variant.graph.node), 2)
        original_result = ort.InferenceSession(original.SerializeToString()).run(None, self.inputs)[0]
        variant_result = ort.InferenceSession(variant.SerializeToString()).run(None, self.inputs)[0]
        self.assertFalse(np.array_equal(original_result, variant_result))
        np.testing.assert_array_equal(variant_result, [[1.5, 2.0, 3.0, 5.5]])

    def test_rejects_incompatible_qdq_without_partial_rewrite(self):
        model = ir.serde.deserialize_model(make_model())
        pairs = activation_pairs(model.graph)
        pairs[1].dequantize.replace_input_with(1, model.graph.initializers["weight_scale"])
        with self.assertRaisesRegex(ValueError, "scale or zero point differs"):
            bypass_pairs(model, pairs)
        self.assertEqual(len(activation_pairs(model.graph)), 2)

    def test_preserves_output_name_when_bypassing_final_qdq(self):
        proto = make_model()
        proto.graph.output[0].name = "b_dq"
        model = ir.serde.deserialize_model(proto)
        selected = select_pairs(activation_pairs(model.graph), re.compile("group_b"), None, [])
        bypass_pairs(model, selected)
        serialized = ir.serde.serialize_model(model)
        self.assertEqual([output.name for output in serialized.graph.output], ["b_dq"])
        result = ort.InferenceSession(serialized.SerializeToString()).run(None, self.inputs)[0]
        np.testing.assert_allclose(result, [[-0.25, 0.25, 1.25, 4.25]], atol=1e-5)

    def test_bypasses_microsoft_domain_qdq_for_qnn_profiles(self):
        model = ir.serde.deserialize_model(make_model("com.microsoft"))
        pairs = activation_pairs(model.graph)
        self.assertEqual(len(pairs), 2)
        bypass_pairs(model, select_pairs(pairs, re.compile("group_a"), None, []))
        result = ort.InferenceSession(ir.serde.serialize_model(model).SerializeToString()).run(None, self.inputs)[0]
        np.testing.assert_array_equal(result, [[1.5, 2.0, 3.0, 5.5]])

    def test_discovers_attention_core_by_graph_topology(self):
        model = ir.serde.deserialize_model(make_attention_model())
        selected = attention_core_pairs(model.graph)
        self.assertEqual([pair.name for pair in selected], ["score_Q", "mask_add_Q", "softmax_Q"])
        bypass_pairs(model, selected)
        inputs = {
            name: np.zeros((1, 2, 2), dtype=np.float32)
            for name in ("query", "keys", "values")
        }
        result = ort.InferenceSession(ir.serde.serialize_model(model).SerializeToString()).run(None, inputs)[0]
        np.testing.assert_array_equal(result, np.zeros((1, 2, 2), dtype=np.float32))

    def test_reuses_external_data_and_preserves_cpu_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.onnx"
            output = root / "variant.onnx"
            onnx.save_model(
                make_model(), source, save_as_external_data=True, all_tensors_to_one_file=True,
                location="source.onnx.data", size_threshold=0,
            )
            model = ir.load(source)
            bypass_pairs(model, select_pairs(activation_pairs(model.graph), re.compile("group_a"), None, []))
            write_variant(model, source, output, 1, {"mode": "bypass", "selected_pairs": ["group_a_Q"]})
            self.assertFalse((root / "variant.onnx.data").exists())
            self.assertEqual(
                json.loads((root / "variant.ablation.json").read_text(encoding="utf-8"))["selected_pairs"],
                ["group_a_Q"],
            )
            self.assertEqual(len({p.key: p.value for p in onnx.load(output, load_external_data=False).metadata_props}[
                "qdq_ablation_selection_sha256"
            ]), 64)
            np.testing.assert_array_equal(
                ort.InferenceSession(str(output)).run(None, self.inputs)[0],
                [[1.5, 2.0, 3.0, 5.5]],
            )

    def test_validates_archived_inputs_and_numeric_metrics(self):
        session = ort.InferenceSession(make_model().SerializeToString())
        outputs, latencies = run_session(session, self.inputs, warmup_iterations=1, iterations=3)
        self.assertEqual(len(latencies), 3)
        self.assertEqual(outputs[0].shape, (1, 4))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inputs.npz"
            np.savez(path, x=self.inputs["x"], valid_mask=np.asarray([[True, False, True, False]]))
            inputs, mask = load_inputs(session, path, "valid_mask")
            np.testing.assert_array_equal(inputs["x"], self.inputs["x"])
            outputs = compare_outputs(
                ["y"], [self.inputs["x"]], [self.inputs["x"] + np.float32(0.1)], 1e-3, 1e-2, mask,
            )
            self.assertAlmostEqual(outputs["y"]["valid_mean_abs_error"], 0.1, places=6)
            np.savez(path, x=self.inputs["x"].astype(np.float16), valid_mask=mask)
            with self.assertRaisesRegex(ValueError, "expected"):
                load_inputs(session, path, "valid_mask")
        self.assertIsNone(
            measure_float_output(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), 0, 0)[
                "relative_l2_error"
            ]
        )

    def test_rejects_compiled_model_with_unclaimed_nodes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "compiled.onnx"
            proto = make_model()
            del proto.graph.node[:]
            proto.graph.node.append(helper.make_node("EPContext", ["x"], ["intermediate"]))
            proto.graph.node.append(helper.make_node("Identity", ["intermediate"], ["y"]))
            onnx.save(proto, path)
            with self.assertRaisesRegex(RuntimeError, "non-EPContext nodes"):
                context_node_count(path)
            del proto.graph.node[1]
            onnx.save(proto, path)
            self.assertEqual(context_node_count(path), 1)

    def test_rejects_compiled_context_for_a_different_variant(self):
        with tempfile.TemporaryDirectory() as temp:
            source_path, context_path = Path(temp) / "source.onnx", Path(temp) / "context.onnx"
            source = make_model()
            source.metadata_props.add(key="qdq_ablation_pairs_removed", value="3")
            context = make_model()
            context.metadata_props.add(key="qdq_ablation_pairs_removed", value="2")
            onnx.save(source, source_path)
            onnx.save(context, context_path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_compiled_source(source_path, context_path)
            context.metadata_props[0].value = "3"
            onnx.save(context, context_path)
            validate_compiled_source(source_path, context_path)


if __name__ == "__main__":
    unittest.main()
