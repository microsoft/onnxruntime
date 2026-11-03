import copy
import unittest

from run_prefill256_model import verify_metadata, verify_tokens
from run_qmoe_prefill256_native import gate_rows, verify_gate_coverage


def gate(shape):
    return dict(cat="Kernel", name="layer&QMoE&QmoeGate1Token",
                args=dict(shapes=f"inputs[0] = {{{shape}}} outputs[0] = {{{shape}}} "))


class CoverageTests(unittest.TestCase):
    def test_batch_and_sequence_dimensions(self):
        self.assertEqual(gate_rows(gate("256,512")), 256)
        self.assertEqual(gate_rows(gate("1,256,512")), 256)
        self.assertEqual(gate_rows(gate("2,128,512")), 256)
        with self.assertRaises(AssertionError):
            gate_rows(gate("256,511"))

    def test_exact_native_coverage(self):
        events = [gate("16,512")] * 208 + [gate("128,512")] * 8 + [gate("256,512")] * 8
        self.assertEqual(verify_gate_coverage(events), {16: 208, 128: 8, 256: 8})
        with self.assertRaises(AssertionError):
            verify_gate_coverage(events[:-1])
        with self.assertRaises(AssertionError):
            verify_gate_coverage(events + [gate("255,512")])


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.references = [dict(prompt_sha256=str(i), output_token_ids=list(range(256))) for i in range(4)]
        self.raw = dict(samples=[
            dict(warmup=i == 0, context=2048, prompt_sha256=str(i),
                 output_token_ids=list(range(36)), output_tokens=36, decode_tps=70.,
                 decode_seconds=.5, request_seconds=6., ttft_seconds=5.5,
                 process_resource_usage=dict(total=dict(major_page_faults=0)))
            for i in range(2)], summary={"2048:configured": dict(decode_tps=70.)})

    def test_frozen_prefix(self):
        self.assertEqual(verify_tokens(self.raw, self.references, 2048, 36), dict(decode_tps=70.))

    def test_rejects_metric_fault_and_token_corruption(self):
        for field, value in (("decode_tps", 71.), ("decode_seconds", float("nan")),
                             ("request_seconds", 0), ("ttft_seconds", float("inf"))):
            raw = copy.deepcopy(self.raw)
            raw["samples"][0][field] = value
            with self.assertRaises(AssertionError):
                verify_tokens(raw, self.references, 2048, 36)
        raw = copy.deepcopy(self.raw)
        raw["samples"][0]["output_token_ids"][0] = -1
        with self.assertRaises(AssertionError):
            verify_tokens(raw, self.references, 2048, 36)
        raw = copy.deepcopy(self.raw)
        raw["samples"][0]["process_resource_usage"]["total"]["major_page_faults"] = 1
        with self.assertRaises(AssertionError):
            verify_tokens(raw, self.references, 2048, 36)

    def test_standard_metrics_exclude_warmup(self):
        samples = []
        for i in range(4):
            sample = copy.deepcopy(self.raw["samples"][0])
            sample.update(warmup=i == 0, prompt_sha256=str(i), output_token_ids=list(range(256)),
                          output_tokens=256, decode_tps=10. if i == 0 else 70.,
                          decode_seconds=255 / (10. if i == 0 else 70.))
            samples.append(sample)
        expected = dict(decode_tps=70., median_decode_tps=70., mean_ttft_seconds=5.5, e2e_tps=256 / 6)
        raw = dict(samples=samples, summary={"2048:configured": expected})
        self.assertEqual(verify_tokens(raw, self.references, 2048, 256), expected)
        raw["summary"]["2048:configured"]["decode_tps"] = 71.
        with self.assertRaises(AssertionError):
            verify_tokens(raw, self.references, 2048, 256)


class MetadataTests(unittest.TestCase):
    def test_metadata_and_selectors(self):
        native = dict(provider_sha256="provider")
        parent = dict(graph_sha256="graph", config_sha256="config")
        raw = dict(status="completed", plugin_sha256="provider", graph_sha256="graph",
                   config_sha256="config", provider="webgpu", chunk=256,
                   webgpu_max_pending_dispatches_override=64, webgpu_validation_mode_override="full",
                   load_experiment_flags={
                       "ORT_WEBGPU_QMOE_PREFILL256": "1", "ORT_WEBGPU_QMOE_EXTENDED_PREFILL": "1",
                       "ORT_WEBGPU_QSA_INDEXER_STACK": "1", "ORT_WEBGPU_DENSE_BITCAST_UNPACK": "1",
                       "ORT_WEBGPU_QSA_CONCAT_VECTOR": "0", "ORT_WEBGPU_QSA_PACKED_SCORE": "0"})
        guard = dict(status="completed", exit_code=0, gpu_index=7,
                     gpu_uuid="GPU-0e70e969-bbfc-80c5-d582-88be134632ea")
        verify_metadata(raw, guard, native, parent, True)
        for field, value in (("plugin_sha256", "wrong"), ("chunk", 128),
                             ("webgpu_max_pending_dispatches_override", 128),
                             ("webgpu_validation_mode_override", None)):
            changed = copy.deepcopy(raw)
            changed[field] = value
            with self.assertRaises(AssertionError):
                verify_metadata(changed, guard, native, parent, True)
        for flag in raw["load_experiment_flags"]:
            changed = copy.deepcopy(raw)
            changed["load_experiment_flags"][flag] = "2"
            with self.assertRaises(AssertionError):
                verify_metadata(changed, guard, native, parent, True)
        raw["webgpu_validation_mode_override"] = None
        verify_metadata(raw, guard, native, parent, False)


if __name__ == "__main__":
    unittest.main()
