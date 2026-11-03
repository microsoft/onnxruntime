from pathlib import Path
import unittest

from build_gdn_prep_node_replay import (
    COMPARATOR, FLAG, HEADER, KERNEL, SOURCE, transform_comparator, transform_header, transform_source)
from prepare_gdn_prep_replay_model import ROOT, base, model_source
from qualify_row4_cpu_locality import compare_samples


def host(path):
    return Path(str(path).replace("/workspace", "/home/sayanshaw/repos", 1))


class GdnPrepNodeReplayTests(unittest.TestCase):
    def test_private_constructor_setting_and_five_bindings(self):
        source = transform_header(host(HEADER).read_text())
        self.assertIn(f"class {KERNEL} final", source)
        self.assertIn(f'ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)', source)
        self.assertIn(f'"{FLAG} must be 0 or 1"', source)
        self.assertIn('"GdnProjPrepF16Exact", {0, 1, 2, 3, 4}', source)

    def test_shader_and_fallback_paths_are_unchanged(self):
        original = host(SOURCE).read_text()
        changed = transform_source(original)
        start = original.index("Status GdnProjPrepF16::ComputeInternal")
        modified_start = changed.index(f"Status {KERNEL}::ComputeInternal")
        self.assertEqual(original[:start].replace("    GdnProjPrepF16);", f"    {KERNEL});"),
                         changed[:modified_start])
        self.assertIn("if (replay_enabled_ && batch == 1 && length == 1)", changed)
        self.assertIn("return replay_.Run(context, {conv, z, a, b, present}, compute);", changed)
        self.assertIn("return context.CopyTensor(*state, *present);", changed)
        self.assertNotIn(FLAG, changed)
        original_program = original[original.index("  GdnProjPrepF16Program program;", start):
                                    original.index("\n}", start)]
        self.assertIn("\n".join("  " + line for line in original_program.splitlines()), changed)

    def test_dual_sessions_all_pending_windows_and_different_live_addresses(self):
        source = transform_comparator(host(COMPARATOR).read_text())
        self.assertIn('flag("0");\n      Ort::Session session', source)
        self.assertIn('flag("1");\n      Ort::Session candidate_session', source)
        self.assertIn("for (int pending : {1, 64})", source)
        self.assertIn("gpu_inputs[0]", source)
        self.assertIn("gpu_inputs[1 + round % 2]", source)
        self.assertIn("env.CopyTensors(inputs, reference_bank, nullptr)", source)
        self.assertIn("env.CopyTensors(inputs, candidate_bank, nullptr)", source)
        self.assertIn("Ort::Session invalid_session", source)
        self.assertIn('{"enableMatmulFp32Accumulation", fp32}', source)

    def test_model_requires_exact_native_and_negative_proofs_and_both_operator_families(self):
        source = model_source((ROOT / "run_ready_replay_key_model.py").read_text())
        self.assertIn('native["finite_bitwise_comparisons"] == 181043200', source)
        self.assertIn('negative["provider_sha256"] == native["provider_sha256"]', source)
        self.assertIn('native["distinct_live_gpu_banks"] == 3', source)
        self.assertIn("assert len(dense) == 171", source)
        self.assertIn("assert len(prep) == 36", source)
        self.assertIn('if args.mode == "control":', source)
        self.assertIn("qwen38-int4-qmoe-prefill-rows4/build-v2/", source)
        self.assertNotIn("key_coverage", source)
        self.assertIn("constructor_static_experiment=True", source)

    def test_screen_preserves_original_provider_and_required_support(self):
        source = base((ROOT / "run_ready_replay_key_screen.py").read_text())
        self.assertIn("0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4", source)
        self.assertIn('providers["control"] != providers["prep"]', source)
        self.assertIn('supported = all(row["comparison"]["supported"] for row in rows)', source)
        self.assertIn("compare_samples(samples[1], samples[0], samples[2])", source)

    def test_measurable_request_and_ttft_support_thresholds(self):
        def samples(rates, ttft=1):
            return [dict(prompt_sha256="same", output_token_ids=[1, 2],
                         decode_tps=rate, ttft_seconds=ttft) for rate in rates]
        before = samples([80] * 3)
        after = samples([125] * 3)
        result = compare_samples(samples([110] * 3), before, after)
        self.assertTrue(result["supported"])
        self.assertAlmostEqual(result["matched_request_decode_ratio"], 1.1)
        self.assertFalse(compare_samples(samples([110, 110, 99]), before, after)["supported"])
        self.assertFalse(compare_samples(samples([110] * 3, 1.021), before, after)["supported"])
        self.assertTrue(compare_samples(samples([110] * 3, 1.02), before, after)["supported"])


if __name__ == "__main__":
    unittest.main()
