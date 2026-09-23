"""Exercise weight-free Gemma operator QDQ models with diagnostic CPU clipping."""

import unittest

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

from drift_investigation.generate_qdq_operator_model import ACTIVATION_TYPES, OPERATORS, REGIMES, build_model
from drift_investigation.qdq_ablation import activation_pairs, bypass_pairs
from run_acc import measure_float_output


class GenerateQdqOperatorModelTest(unittest.TestCase):
    def test_all_operator_and_dtype_combinations_have_measurable_qdq_effect(self):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        for op in OPERATORS:
            for activation_type in ACTIVATION_TYPES:
                for regime in REGIMES:
                    with self.subTest(operator=op, dtype=activation_type, regime=regime):
                        model, inputs = build_model(op, activation_type, regime, 1009)
                        original = ort.InferenceSession(
                            model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
                        ).run(None, inputs)[0]
                        variant = ir.serde.deserialize_model(model)
                        chosen = "QuantizeActivation" if regime == "input_clipped" else "QuantizeOutput"
                        pairs = activation_pairs(variant.graph)
                        self.assertEqual(len(pairs), 2)
                        bypass_pairs(variant, [pair for pair in pairs if pair.name == chosen])
                        bypassed = ort.InferenceSession(
                            ir.serde.serialize_model(variant).SerializeToString(),
                            sess_options=options, providers=["CPUExecutionProvider"],
                        ).run(None, inputs)[0]
                        self.assertGreater(measure_float_output(original, bypassed, 0, 0)["mean_abs_error"], 0.001)
                        self.assertTrue(np.isfinite(original).all())

    def test_invalid_operator_and_activation_configurations_fail(self):
        with self.assertRaisesRegex(ValueError, "unsupported operator"):
            build_model("Attention", "uint16", "input_clipped", 0)
        with self.assertRaisesRegex(ValueError, "unsupported activation"):
            build_model("Add", "int8", "input_clipped", 0)
        with self.assertRaisesRegex(ValueError, "unsupported activation"):
            build_model("Add", "uint16", "missing_regime", 0)


if __name__ == "__main__":
    unittest.main()
