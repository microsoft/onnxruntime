import collections
import json
import unittest

from run_dense_lane8_model import DIRECTORY, ROOT, TEMPLATE, model_command, prefix_for, verify_profile


class DenseLane8ModelTests(unittest.TestCase):
    def setUp(self):
        self.template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        self.provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"

    def test_standard_controls_and_candidate_keep_worker_policy(self):
        for mode in ("control", "lane8"):
            for full in (False, True):
                for context in (2048, 8192, 32768):
                    with self.subTest(mode=mode, full=full, context=context):
                        prefix = prefix_for(mode, context, full, 0)
                        command = model_command(self.template, self.provider, prefix, context, full, mode, "standard")
                        self.assertIn(f"export ORT_WEBGPU_DENSE_LANE8={int(mode == 'lane8')}; ", command[-1])
                        self.assertIn(f"--contexts {context}", command[-1])
                        self.assertIn("--tokens 256 --repetitions 3", command[-1])
                        self.assertIn("ORT_WEBGPU_QMOE_GROUPED_PREFILL=1", command[-1])
                        self.assertIn("ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD=0", command[-1])
                        self.assertEqual(" --webgpu-validation-mode full" in command[-1], full)
                        self.assertNotIn("--inherit-worker-cpu-affinity", command[-1])
                        self.assertNotIn("taskset", command)
                        self.assertNotIn("--compare-", command[-1])
                        self.assertNotIn("--profile-", command[-1])

    def test_profile_has_one_decode_token_and_distinct_programs(self):
        prefix = prefix_for("lane8", 2048, True, 0, "profile")
        command = model_command(self.template, self.provider, prefix, 2048, True, "lane8", "profile")
        self.assertIn("--tokens 36 --repetitions 1 --profile-decode-after 35", command[-1])
        for mode in ("control", "lane8"):
            counts = collections.Counter(DynamicSparseAttentionQueryCache=12)
            counts["MatMulNBitsLane8F16" if mode == "lane8" else "MatMulNBitsBlock32F16"] = 159
            verify_profile(counts, mode)
            counts["MatMulNBitsTile4F16"] = 1
            with self.assertRaises(AssertionError):
                verify_profile(counts, mode)
        with self.assertRaises(AssertionError):
            model_command(self.template, self.provider, prefix, 2048, False, "lane8", "profile")

    def test_profile_rejects_missing_or_old_dispatch(self):
        for counts in (collections.Counter(MatMulNBitsLane8F16=158, DynamicSparseAttentionQueryCache=12),
                       collections.Counter(MatMulNBitsLane8F16=159, MatMulNBitsBlock32F16=1,
                                           DynamicSparseAttentionQueryCache=12),
                       collections.Counter(MatMulNBitsLane8F16=159, DynamicSparseAttentionQueryCache=11)):
            with self.subTest(counts=counts), self.assertRaises(AssertionError):
                verify_profile(counts, "lane8")

    def test_profile_and_trials_have_disjoint_paths(self):
        paths = {prefix_for(mode, 2048, full, trial, phase)
                 for mode in ("control", "lane8") for full in (False, True)
                 for trial in (0, 1, 2) for phase in ("profile", "standard")}
        self.assertEqual(len(paths), 24)


if __name__ == "__main__":
    unittest.main()
