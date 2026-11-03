import copy
import unittest

from build_qsa_score16 import FLAG
from qualify_qsa_score16_standard import all_contexts_reach_target, verify_standard


class StandardScore16Tests(unittest.TestCase):
    def setUp(self):
        self.native = dict(provider_sha256="provider")
        self.parent = dict(graph_sha256="graph", config_sha256="config")
        self.guard = dict(status="completed", exit_code=0, gpu_index=7,
                          gpu_uuid="GPU-0e70e969-bbfc-80c5-d582-88be134632ea")
        self.references = [dict(prompt_sha256=str(i), output_token_ids=list(range(256))) for i in range(4)]
        samples = []
        for i in range(4):
            tps = 20. if i == 0 else 100.
            samples.append(dict(context=2048, warmup=i == 0, prompt_sha256=str(i),
                output_tokens=256, output_token_ids=list(range(256)),
                decode_tps=tps, decode_seconds=255 / tps, request_seconds=1 + 255 / tps,
                ttft_seconds=1., process_resource_usage=dict(total=dict(major_page_faults=0))))
        self.expected = dict(decode_tps=100., median_decode_tps=100.,
                             mean_ttft_seconds=1., e2e_tps=256 / 3.55)
        self.raw = dict(status="completed", plugin_sha256="provider", graph_sha256="graph",
            config_sha256="config", provider="webgpu", chunk=256,
            webgpu_max_pending_dispatches_override=64, webgpu_validation_mode_override="full",
            profiling=False, profiling_scope=None, profile_decode_after_tokens=None, comparison=None,
            load_experiment_flags={FLAG: "1", "ORT_WEBGPU_QMOE_PREFILL256": "1",
                "ORT_WEBGPU_QMOE_EXTENDED_PREFILL": "1", "ORT_WEBGPU_QSA_INDEXER_STACK": "1",
                "ORT_WEBGPU_DENSE_BITCAST_UNPACK": "1", "ORT_WEBGPU_QSA_CONCAT_VECTOR": "0",
                "ORT_WEBGPU_QSA_PACKED_SCORE": "0"},
            samples=samples, summary={"2048:configured": self.expected})

    def verify(self, raw, full=True):
        return verify_standard(raw, self.guard, self.native, self.parent, self.references, 2048, full)

    def test_exact_metrics_and_validation_modes(self):
        self.assertEqual(self.verify(self.raw), self.expected)
        raw = copy.deepcopy(self.raw)
        raw["webgpu_validation_mode_override"] = None
        self.assertEqual(self.verify(raw, False), self.expected)

    def test_rejects_profiling_and_wrong_selector(self):
        for key, value in (("profiling", True), ("profiling_scope", "decode_only"),
                           ("comparison", "qsa_score16_decode_windows"), ("profile_decode_after_tokens", 35)):
            raw = copy.deepcopy(self.raw)
            raw[key] = value
            with self.assertRaises(AssertionError):
                self.verify(raw)
        raw = copy.deepcopy(self.raw)
        raw["load_experiment_flags"][FLAG] = "0"
        with self.assertRaises(AssertionError):
            self.verify(raw)

    def test_rejects_metric_and_warmup_fault_corruption(self):
        raw = copy.deepcopy(self.raw)
        raw["samples"][1]["decode_tps"] = 101.
        with self.assertRaises(AssertionError):
            self.verify(raw)
        raw = copy.deepcopy(self.raw)
        raw["samples"][0]["process_resource_usage"]["total"]["major_page_faults"] = 1
        with self.assertRaises(AssertionError):
            self.verify(raw)

    def test_target_is_exact_and_per_context(self):
        rows = [dict(context=c, candidate=dict(median_decode_tps=100.)) for c in (2048, 8192, 32768)]
        self.assertTrue(all_contexts_reach_target(rows))
        rows[2]["candidate"]["median_decode_tps"] = 99.999999
        self.assertFalse(all_contexts_reach_target(rows))
        rows[0]["candidate"]["median_decode_tps"] = 1000.
        self.assertFalse(all_contexts_reach_target(rows))
        with self.assertRaises(AssertionError):
            all_contexts_reach_target(rows[:-1])


if __name__ == "__main__":
    unittest.main()
