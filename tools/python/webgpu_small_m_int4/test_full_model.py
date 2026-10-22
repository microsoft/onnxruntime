# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for full-model accounting, route verification, and isolated instrumentation."""

import json
import tempfile
import unittest
from pathlib import Path

from build_full_model import instrument_provider
from full_model import profile_summary, stats_delta

from build import ROOT, instrument_source


def counters():
    return dict.fromkeys(
        (
            "rounds",
            "completed_rounds",
            "draft_tokens_proposed",
            "draft_tokens_evaluated",
            "draft_tokens_accepted",
            "target_forward_passes",
            "draft_forward_passes",
            "dflash2_failures",
            "dflash2_disables",
            "dflash2_admission_misses",
            "standard_fallback_steps",
        ),
        0,
    ) | {"acceptance_length_histogram": [0, 0, 0, 0]}


class FullModelTest(unittest.TestCase):
    def test_speculative_counts_and_acceptance(self):
        before, after = counters(), counters()
        after.update(
            rounds=2,
            completed_rounds=2,
            draft_tokens_proposed=6,
            draft_tokens_evaluated=5,
            draft_tokens_accepted=3,
            target_forward_passes=2,
            draft_forward_passes=2,
            acceptance_length_histogram=[0, 1, 1, 0],
        )
        result = stats_delta(before, after, True)
        self.assertEqual(result["acceptance_rate"], 0.6)
        self.assertEqual(result["accepted_per_verification"], 1.5)
        self.assertEqual(result["acceptance_length_histogram"], [0, 1, 1, 0])

    def test_baseline_and_failed_speculation_are_not_false_green(self):
        self.assertIsNone(stats_delta(counters(), counters(), False)["acceptance_rate"])
        with self.assertRaises(AssertionError):
            stats_delta(counters(), counters(), True)
        after = counters()
        after["dflash2_failures"] = 1
        with self.assertRaises(AssertionError):
            stats_delta(counters(), after, False)

    def test_invalid_acceptance_is_rejected(self):
        after = counters()
        after.update(draft_tokens_proposed=1, draft_tokens_evaluated=1, draft_tokens_accepted=2)
        with self.assertRaises(AssertionError):
            stats_delta(counters(), after, True)

    def test_actual_simt_profile_geometry_and_accounting(self):
        event = {
            "name": "MatMulNBitsMlp&MatMulNBitsMlp&DiagnosticSmallMSimt",
            "dur": 137,
            "shapes": "inputs[1] = {17408,160,1} outputs[0] = {4,17408}",
            "cache_key": "DiagnosticSmallMSimt[4|8|512|4|1|0|0|0]",
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.jsonl"
            path.write_text(json.dumps({"kernels": [event, event]}) + "\n")
            result = profile_summary(path)
            self.assertEqual(result["simt_routes"], {"M4_K5120_N17408": 2})
            self.assertEqual(result["mlp_gpu_ms"], {"gate": 0.137, "up": 0.137})
            self.assertEqual(result["gpu_ms"], 0.274)
            event["shapes"] = "inputs[1] = {1024,160,1} outputs[0] = {4,1024}"
            path.write_text(json.dumps({"kernels": [event]}) + "\n")
            with self.assertRaises(AssertionError):
                profile_summary(path)

    def test_provider_instrumentation_is_isolated_from_graph_replay(self):
        original = (ROOT / "onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc").read_text()
        changed = instrument_provider(original)
        self.assertEqual(changed.count("DiagnosticGpuProfilePath()"), 2)
        self.assertEqual(changed.count("WriteDiagnosticGpuProfile(this,"), 1)
        replay = original.index("WebGpuExecutionProvider::ReplayGraph(")
        self.assertIn(original[replay:], changed)

    def test_explicit_diagnostic_modes_precede_integrated_routing(self):
        original = (ROOT / "onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc").read_text()
        changed = instrument_source(original)
        self.assertLess(
            changed.index('if (diagnostic_mode == "simt")'),
            changed.index("if (!diagnostic_regular && experimental_simt"),
        )
        self.assertIn("!diagnostic_regular && CanApplySubgroupMatrixMatMulNBits(", changed)


if __name__ == "__main__":
    unittest.main()
