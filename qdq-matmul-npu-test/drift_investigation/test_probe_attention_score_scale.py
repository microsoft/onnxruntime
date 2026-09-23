"""Verify that the attention score-scale probe distinguishes causal and noncausal outputs."""

import unittest

import numpy as np
import onnx_ir as ir

from drift_investigation.probe_attention_score_scale import attention_reference, probe, quant_params
from drift_investigation.qdq_ablation import attention_core_groups
from drift_investigation.test_qdq_ablation import make_attention_model


class ProbeAttentionScoreScaleTest(unittest.TestCase):
    def test_exact_qdq_reference_identifies_original_scale(self):
        model = ir.serde.deserialize_model(make_attention_model())
        model.graph.initializers["scale"].const_value = ir.tensor(np.asarray(0.1, dtype=np.float32))
        model.graph.initializers["zp"].const_value = ir.tensor(np.asarray(128, dtype=np.uint8))
        _, *pairs = attention_core_groups(model.graph)[0]
        parameters = tuple(quant_params(pair) for pair in pairs)
        inputs = {
            "query": np.asarray([[[[1, 0], [0, 1]]]], dtype=np.float32),
            "key": np.asarray([[[[1, 0], [0, 1]]]], dtype=np.float32),
            "value": np.asarray([[[[1, 0], [0, 1]]]], dtype=np.float32),
            "attention_mask": np.zeros((1, 1, 1, 2), dtype=np.float32),
        }
        indices = np.arange(2)
        output = attention_reference(
            inputs["query"], inputs["key"], inputs["value"], inputs["attention_mask"],
            indices, 1.0, parameters,
        )
        result = probe(model, inputs, output, [0.5, 1.0], 2, output)
        self.assertEqual(result["best_factor"], 1.0)
        self.assertEqual(result["cpu_reference_mae"], 0.0)
        self.assertEqual(result["last_query_causal_effect"], 0.0)
        self.assertGreater(result["causal_mae"], 0.0)
        with self.assertRaisesRegex(ValueError, "include 1.0"):
            probe(model, inputs, output, [0.5], 2)


if __name__ == "__main__":
    unittest.main()
