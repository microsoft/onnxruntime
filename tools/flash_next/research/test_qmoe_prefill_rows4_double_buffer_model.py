import collections
import json
import unittest

from build_qmoe_prefill_rows4_double_buffer import FLAG, NAME
from prepare_qmoe_prefill_rows4_double_buffer_model import benchmark_source, ROOT
from run_qmoe_prefill_rows4_double_buffer_model import model_command, prefix_for, verify_profile
from run_qmoe_prefill_rows4_standard import TEMPLATE


class PrefillModelTests(unittest.TestCase):
    def test_benchmark_changes_only_flag_recording(self):
        self.assertEqual(benchmark_source().replace(f'                "{FLAG}",\n', ""),
                         (ROOT / "benchmark_webgpu_int4.py").read_text())

    def test_all_context_modes_commands_and_profile_counts(self):
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            expected = 96 * (context // 1024)
            counts = collections.Counter({NAME: expected, "MatMulNBitsGroupedRows4F16": expected})
            verify_profile(counts, context)
            with self.assertRaises(AssertionError):
                verify_profile(collections.Counter({NAME: expected + 1}), context)
            for full in (True, False):
                for phase in ("profile", "screen"):
                    for mode in (0, 1):
                        prefix = prefix_for(context, full, phase, mode, 0)
                        provider = ROOT / "test-provider.so"
                        command = model_command(template, provider, prefix, context, full, phase, mode)
                        self.assertIn(f"export {FLAG}={mode};", command[-1])
                        self.assertIn("export ORT_WEBGPU_QMOE_GROUPED_PREFILL=1", command[-1])
                        self.assertIn(f"--contexts {context}", command[-1])
                        self.assertIn("--tokens 2 --repetitions 1" if phase == "profile" else
                                      "--tokens 36 --repetitions 1", command[-1])
                        self.assertEqual("--webgpu-validation-mode full" in command[-1], full)
                        self.assertEqual("--profile-prefix" in command[-1], phase == "profile")
                        self.assertIn("benchmark_webgpu_int4_prefill_double_buffer.py", command[-1])


if __name__ == "__main__":
    unittest.main()
