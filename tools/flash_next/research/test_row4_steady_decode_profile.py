import copy
import json
from pathlib import Path
import unittest

from profile_row4_steady_decode import DIRECTORY, ROOT, TEMPLATE, profile_command, verify_profile_samples


class Row4SteadyDecodeProfileTests(unittest.TestCase):
    def test_profiles_exactly_final_decode_with_original_provider(self):
        template = json.loads((ROOT / (TEMPLATE + ".guard.json")).read_text())["command"]
        command = profile_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                  ROOT / "fixture-steady", 32768)[-1]
        self.assertIn("--tokens 256 --repetitions 1", command)
        self.assertIn("--profile-decode-after 255", command)
        self.assertIn("--contexts 32768", command)
        self.assertIn("qwen38-int4-qmoe-prefill-rows4/build-v2/", command)
        self.assertIn(" --webgpu-validation-mode full", command)

    def test_checks_512_ids_and_exact_profile_boundary(self):
        reference = dict(prompt_sha256="prompt", output_token_ids=list(range(256)))
        sample = dict(context=32768, prompt_sha256="prompt", output_tokens=256,
            output_token_ids=list(range(256)), profile_after_tokens=255, profiled_decode_tokens=1,
            process_resource_usage={"prefill": {"major_page_faults": 0}, "decode": {"major_page_faults": 0}},
            decode_seconds=3, decode_tps=85, request_seconds=63, ttft_seconds=60)
        raw = dict(samples=[dict(sample, warmup=True), dict(sample, warmup=False)])
        verify_profile_samples(raw, [reference, reference], 32768)
        for field, value in (("profile_after_tokens", 254), ("profiled_decode_tokens", 2),
                             ("output_tokens", 255), ("output_token_ids", list(range(255)))):
            changed = copy.deepcopy(raw)
            changed["samples"][1][field] = value
            with self.assertRaises(AssertionError):
                verify_profile_samples(changed, [reference, reference], 32768)


if __name__ == "__main__":
    unittest.main()
