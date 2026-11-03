import copy
import unittest

from qualify_prefill128_dense import verify_samples


class StandardSamplesTests(unittest.TestCase):
    def fixture(self):
        samples = []
        for index in range(4):
            samples.append(dict(context=2048, warmup=index == 0, prompt_sha256=f"prompt{index}",
                output_token_ids=list(range(256)), output_tokens=256, decode_tps=85,
                decode_seconds=3, request_seconds=8 if index else 103, ttft_seconds=5 if index else 100,
                process_resource_usage=dict(prefill=dict(major_page_faults=0), decode=dict(major_page_faults=0))))
        return dict(samples=samples, summary={"2048:configured": dict(
            median_decode_tps=85, decode_tps=85, e2e_tps=32, mean_ttft_seconds=5)}), copy.deepcopy(samples)

    def test_excludes_warmup_and_recomputes_all_metrics(self):
        raw, references = self.fixture()
        self.assertEqual(verify_samples(raw, references, 2048)["mean_ttft_seconds"], 5)

    def test_rejects_late_divergence(self):
        raw, references = self.fixture()
        raw["samples"][-1]["output_token_ids"][-1] = -1
        with self.assertRaises(AssertionError):
            verify_samples(raw, references, 2048)

    def test_rejects_any_major_fault(self):
        raw, references = self.fixture()
        raw["samples"][0]["process_resource_usage"]["prefill"]["major_page_faults"] = 1
        with self.assertRaises(AssertionError):
            verify_samples(raw, references, 2048)

    def test_rejects_incorrect_summary(self):
        raw, references = self.fixture()
        raw["summary"]["2048:configured"]["mean_ttft_seconds"] = 28.75
        with self.assertRaises(AssertionError):
            verify_samples(raw, references, 2048)


if __name__ == "__main__":
    unittest.main()
