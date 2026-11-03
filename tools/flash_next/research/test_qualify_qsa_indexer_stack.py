import copy
import unittest

from qualify_qsa_indexer_stack import UUID, meets_decode_target, verify_standard
import test_qualify_prefill128_dense as parent_tests


class StackQualificationTests(unittest.TestCase):
    def fixture(self):
        raw, references = parent_tests.StandardSamplesTests().fixture()
        raw.update(status="completed", plugin_sha256="provider", graph_sha256="graph",
                   config_sha256="config", provider="webgpu", chunk=128,
                   webgpu_max_pending_dispatches_override=64, webgpu_validation_mode_override="full",
                   load_experiment_flags=dict(ORT_WEBGPU_QSA_INDEXER_STACK="1",
                       ORT_WEBGPU_QSA_CONCAT_VECTOR="0", ORT_WEBGPU_QSA_PACKED_SCORE="0",
                       ORT_WEBGPU_DENSE_BITCAST_UNPACK="1", ORT_WEBGPU_QMOE_EXTENDED_PREFILL="1"))
        proof = dict(status="original_sequences_qualified", provider_sha256="provider",
                     graph_sha256="graph", chunk=128, context=2048, phase="standard",
                     candidate="indexer-stack", full_validation=True,
                     original_output_prefixes_match=True, zero_major_faults=True)
        guard = dict(status="completed", exit_code=0, gpu_index=7, gpu_uuid=UUID)
        native = dict(provider_sha256="provider")
        parent = dict(graph_sha256="graph", config_sha256="config")
        controls = dict(samples=references, config_sha256="config")
        return [raw, guard, proof, native, parent, controls, 2048, True]

    def test_recomputes_joint_metrics_without_warmup(self):
        args = self.fixture()
        self.assertEqual(verify_standard(*args)["mean_ttft_seconds"], 5)

    def test_rejects_mislabeled_decode_rate_even_with_matching_summary(self):
        args = self.fixture()
        for sample in args[0]["samples"]:
            sample["decode_tps"] = 100
        args[0]["summary"]["2048:configured"]["median_decode_tps"] = 100
        with self.assertRaises(AssertionError):
            verify_standard(*args)

    def test_rejects_unmanaged_child_flag_or_wrong_validation_mode(self):
        for change in ("child", "validation", "gpu", "provider"):
            args = self.fixture()
            if change == "child":
                args[0]["load_experiment_flags"]["ORT_WEBGPU_QSA_PACKED_SCORE"] = "1"
            elif change == "validation":
                args[0]["webgpu_validation_mode_override"] = None
            elif change == "gpu":
                args[1]["gpu_index"] = 0
            else:
                args[2]["provider_sha256"] = "different"
            with self.assertRaises(AssertionError):
                verify_standard(*args)

    def test_target_is_per_context_and_includes_exact_100(self):
        rows = [dict(context=c, candidate=dict(median_decode_tps=100)) for c in (2048, 8192, 32768)]
        self.assertTrue(meets_decode_target(rows))
        below = copy.deepcopy(rows)
        below[-1]["candidate"]["median_decode_tps"] = 99.999999
        self.assertFalse(meets_decode_target(below))
        with self.assertRaises(AssertionError):
            meets_decode_target(rows[:2])
        rows[-1]["candidate"]["median_decode_tps"] = float("inf")
        with self.assertRaises(AssertionError):
            meets_decode_target(rows)


if __name__ == "__main__":
    unittest.main()
