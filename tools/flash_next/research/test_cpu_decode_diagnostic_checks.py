import copy
import json
import unittest

from cpu_decode_diagnostic_checks import verify_two_requests
from prepare_webgpu_int4_metrics import ROOT


class CpuDecodeDiagnosticChecksTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "gpu-slice-combined32k-v1-context32768-retained-flag1-trial0-trace0-gpu7.json"
        baseline = json.loads(path.read_text())
        self.references = baseline["samples"]
        self.raw = {"samples": copy.deepcopy(self.references[:2])}

    def test_complete_real_warmup_and_measured_prefixes(self):
        verify_two_requests(self.raw, self.references, 32768)

    def test_rejects_missing_request_and_incomplete_ids(self):
        self.raw["samples"][1]["output_token_ids"].pop()
        with self.assertRaises(RuntimeError):
            verify_two_requests(self.raw, self.references, 32768)
        self.raw["samples"].pop()
        with self.assertRaises(RuntimeError):
            verify_two_requests(self.raw, self.references, 32768)

    def test_rejects_major_faults_and_wrong_denominator(self):
        self.raw["samples"][1]["decode_tps"] *= 2
        with self.assertRaises(RuntimeError):
            verify_two_requests(self.raw, self.references, 32768)
        self.raw = {"samples": copy.deepcopy(self.references[:2])}
        self.raw["samples"][1]["process_resource_usage"]["decode"]["major_page_faults"] = 1
        with self.assertRaises(RuntimeError):
            verify_two_requests(self.raw, self.references, 32768)


if __name__ == "__main__":
    unittest.main()
