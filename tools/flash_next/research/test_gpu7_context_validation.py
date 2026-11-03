import copy
import json
from pathlib import Path
import tempfile
import unittest

from resume_gpu7_8k import GPU_UUID, validate_gpu7


class Gpu7ContextValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "benchmark.json"
        self.identity = dict(provider_sha256="provider", graph_sha256="graph", config_sha256="config")
        samples = []
        for repetition in range(-1, 3):
            samples.append(dict(context=8192, repetition=repetition, warmup=repetition == -1,
                                seed=100 + repetition, offset=10 + repetition, input_tokens=8192,
                                prompt_sha256=f"prompt{repetition}", output_token_ids=list(range(256)),
                                output_sha256="output", output_tokens=256, decode_seconds=5.0,
                                decode_tps=51.0, process_resource_usage={
                                    "prefill": {"major_page_faults": 0},
                                    "decode": {"major_page_faults": 0}}))
        self.control = dict(samples=copy.deepcopy(samples))
        self.data = dict(status="completed", provider="webgpu", chunk=16, profiling=False,
                         webgpu_max_pending_dispatches_override=64,
                         quality_qualified=False, headline_eligible=False,
                         plugin_sha256="provider", graph_sha256="graph", config_sha256="config",
                         samples=samples, summary={"8192:configured": {
                             "median_decode_tps": 51.0, "decode_tps": 51.0}})
        self.guard = dict(status="completed", exit_code=0, gpu_index=7, gpu_uuid=GPU_UUID,
                          command=["ORT_WEBGPU_HC_EXACT_DENSE=1 ORT_WEBGPU_EXACT_DENSE_MID=1"])

    def invoke(self):
        self.path.write_text(json.dumps(self.data))
        self.path.with_suffix(".guard.json").write_text(json.dumps(self.guard))
        self.path.with_suffix(".log").write_text("completed\n")
        return validate_gpu7(self.path, 8192, self.control, self.identity)

    def test_complete_row_qualifies(self):
        self.assertEqual(self.invoke()["original_sequences_exact_including_warmup"], 4)

    def test_wrong_device_rejected(self):
        for key, value in (("gpu_index", 0), ("gpu_uuid", "different-gpu")):
            with self.subTest(key=key):
                original = self.guard[key]
                self.guard[key] = value
                with self.assertRaises(AssertionError):
                    self.invoke()
                self.guard[key] = original

    def test_warmup_token_change_rejected(self):
        self.data["samples"][0]["output_token_ids"][0] = -1
        with self.assertRaises(AssertionError):
            self.invoke()

    def test_incomplete_row_rejected(self):
        self.data["samples"].pop()
        with self.assertRaises(AssertionError):
            self.invoke()

    def test_major_fault_rejected(self):
        self.data["samples"][1]["process_resource_usage"]["decode"]["major_page_faults"] = 1
        with self.assertRaises(AssertionError):
            self.invoke()

    def test_wrong_interval_count_rejected(self):
        self.data["samples"][1]["decode_tps"] = 256 / 5.0
        with self.assertRaises(AssertionError):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
