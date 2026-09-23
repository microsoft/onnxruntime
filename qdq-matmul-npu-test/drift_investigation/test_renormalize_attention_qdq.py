"""Check reversible score-QDQ scaling preserves CPU attention semantics."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx import numpy_helper

from drift_investigation.renormalize_attention_qdq import renormalize, verify_cpu
from drift_investigation.test_qdq_ablation import make_attention_model


class RenormalizeAttentionQdqTest(unittest.TestCase):
    def test_shared_scale_is_cloned_and_basic_cpu_output_is_identical(self):
        model = ir.serde.deserialize_model(make_attention_model())
        model.graph.sort()
        original = ir.serde.serialize_model(model)
        transformed, selected = renormalize(model)
        self.assertEqual(selected, ["attention_softmax"])
        initializers = {value.name: numpy_helper.to_array(value) for value in transformed.graph.initializer}
        self.assertEqual(initializers["scale"].item(), np.float32(1 / 255).item())
        self.assertEqual(
            initializers["diagnostic_attention_scale_0"].item(),
            np.float32(2 * np.float32(1 / 255)).item(),
        )
        self.assertEqual(sum(node.op_type == "Mul" for node in transformed.graph.node), 2)

        inputs = {
            "query": np.asarray([[[0.3, 0.1], [0.1, 0.3]]], dtype=np.float32),
            "keys": np.asarray([[[1, 0], [0, 1]]], dtype=np.float32),
            "values": np.asarray([[[1, 0], [0, 1]]], dtype=np.float32),
        }
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        before = ort.InferenceSession(original.SerializeToString(), sess_options=options).run(None, inputs)[0]
        after = ort.InferenceSession(transformed.SerializeToString(), sess_options=options).run(None, inputs)[0]
        np.testing.assert_array_equal(before, after)

    def test_unknown_softmax_is_rejected_without_mutation(self):
        model = ir.serde.deserialize_model(make_attention_model())
        with self.assertRaisesRegex(ValueError, "unknown QDQ-wrapped"):
            renormalize(model, {"missing_softmax"})
        self.assertFalse(any(name.startswith("diagnostic_") for name in model.graph.initializers))

    def test_verification_rejects_cpu_optimizer_semantic_change(self):
        model = ir.serde.deserialize_model(make_attention_model())
        model.graph.sort()
        original = ir.serde.serialize_model(model)
        changed, _ = renormalize(model)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original_path, changed_path = root / "original.onnx", root / "changed.onnx"
            onnx.save(original, original_path)
            onnx.save(changed, changed_path)
            safe = root / "safe.npz"
            np.savez(safe, **{
                name: np.zeros((1, 2, 2), dtype=np.float32)
                for name in ("query", "keys", "values")
            })
            verify_cpu(original_path, changed_path, [safe], None)
            sensitive = root / "sensitive.npz"
            np.savez(
                sensitive,
                query=np.asarray([[[0.3, 0.1], [0.1, 0.3]]], dtype=np.float32),
                keys=np.asarray([[[1, 0], [0, 1]]], dtype=np.float32),
                values=np.asarray([[[1, 0], [0, 1]]], dtype=np.float32),
            )
            with self.assertRaisesRegex(RuntimeError, "CPU output .* changed"):
                verify_cpu(original_path, changed_path, [sensitive], None)


if __name__ == "__main__":
    unittest.main()
