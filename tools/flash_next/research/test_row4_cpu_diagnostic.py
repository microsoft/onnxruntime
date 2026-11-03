import json
import io
from pathlib import Path
import time
import unittest

from run_row4_cpu_diagnostic import diagnostic_command
from benchmark_row4_cpu_diagnostic import profile_request
from run_qmoe_prefill_rows4_standard import TEMPLATE


class CpuDiagnosticTests(unittest.TestCase):
    def test_instrumented_longer_decode_keeps_frozen_parent_and_output_paths(self):
        root = Path(__file__).parent
        template = json.loads((root / f"{TEMPLATE}.guard.json").read_text())["command"]
        prefix = root / "cpu-diagnostic-test"
        command = diagnostic_command(template, Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/"
                                                     "libonnxruntime_providers_webgpu.so"), prefix)
        self.assertIn("--tokens 768 --repetitions 1", command[-1])
        self.assertIn("ORT_WEBGPU_QMOE_GROUPED_PREFILL=1", command[-1])
        self.assertNotIn(" --webgpu-validation-mode full", command[-1])
        self.assertNotIn("--compare-", command[-1])
        self.assertEqual(command[-1].count(prefix.name), 3)
        self.assertIn("/benchmark_row4_cpu_diagnostic.py", command[-1])
        self.assertNotIn("/benchmark_webgpu_int4.py", command[-1])

    def test_decode_boundaries_exclude_prefill_and_preserve_result(self):
        stream = io.StringIO()
        result = dict(decode_seconds=2.0)

        def original(*args, on_decode_step):
            now = time.perf_counter()
            on_decode_step(0, now + 1)
            on_decode_step(767, now + 3)
            return result

        self.assertIs(profile_request(original, stream, None, None, None, [1] * 2048, 768, 1024), result)
        phase = json.loads(stream.getvalue())
        self.assertEqual(phase["context"], 2048)
        self.assertEqual(phase["decode_end"] - phase["decode_start"], 2.0)

    def test_other_request_routes_are_rejected(self):
        with self.assertRaises(ValueError):
            profile_request(None, None, None, None, None, [], 768, 1024, profile_after_tokens=1)


if __name__ == "__main__":
    unittest.main()
