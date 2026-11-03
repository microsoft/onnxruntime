import json
from pathlib import Path
import unittest

from run_row4_pending_standard import pending_command
from run_qmoe_prefill_rows4_standard import DIRECTORY, TEMPLATE


class PendingStandardTests(unittest.TestCase):
    def test_exact_pending_parameter_retains_provider_chunk_and_request_shape(self):
        root = Path(__file__).parent
        template = json.loads((root / f"{TEMPLATE}.guard.json").read_text())["command"]
        for pending in (8, 16, 32, 64):
            for full in (False, True):
                command = pending_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                          root / "pending-test", 2048, full, pending)
                self.assertEqual(command[-1].count(f"--webgpu-max-pending-dispatches {pending}"), 1)
                self.assertIn("--tokens 256 --repetitions 3", command[-1])
                self.assertIn("--chunk 1024", command[-1])
                self.assertIn("ORT_WEBGPU_QMOE_GROUPED_PREFILL=1", command[-1])
                self.assertEqual(" --webgpu-validation-mode full" in command[-1], full)
                self.assertNotIn("--compare-", command[-1])

    def test_unsupported_parameter_is_rejected(self):
        with self.assertRaises(AssertionError):
            pending_command([], None, None, 2048, True, 0)


if __name__ == "__main__":
    unittest.main()
