import json
import unittest

from run_genai_allocator_standard import FLAG, ROOT, RUNTIME, TEMPLATE, allocator_command, prefix_for
from run_qmoe_prefill_rows4_standard import DIRECTORY


class GenaiAllocatorStandardTests(unittest.TestCase):
    def test_each_context_preserves_standard_sampling_and_only_switches_allocator_flag(self):
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (True, False):
                commands = [allocator_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                               prefix_for(mode, context, full, 0), context, full, mode)
                            for mode in ("control", "single")]
                for mode, command in zip(("control", "single"), commands):
                    self.assertIn(f"export {FLAG}={int(mode == 'single')}; ", command[-1])
                    self.assertIn("--tokens 256 --repetitions 3", command[-1])
                    self.assertIn(str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace"), command[-1])
                    self.assertNotIn("--inherit-worker-cpu-affinity", command[-1])
                    self.assertNotIn("ORT_WEBGPU_DENSE_PACKED4", command[-1])
                    self.assertNotIn("ORT_WEBGPU_DENSE_LANE8", command[-1])


if __name__ == "__main__":
    unittest.main()
