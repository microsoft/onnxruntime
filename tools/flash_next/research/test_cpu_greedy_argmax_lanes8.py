import json
from pathlib import Path
import unittest

from build_cpu_greedy_argmax_lanes8 import ARGMAX, FLAG, SELECTOR, transform_source


SOURCE = Path("/home/sayanshaw/repos/onnxruntime-genai-webgpu-fp8-checkpoint/src/search.cpp")


class CpuGreedyArgMaxTests(unittest.TestCase):
    def test_only_greedy_float_selection_changes(self):
        original = SOURCE.read_text()
        candidate = transform_source(original)
        candidate = candidate.replace('#include "models/env_utils.h"\n', "")
        candidate = candidate.replace("namespace {\n" + ARGMAX + SELECTOR + "}\n", "")
        candidate = candidate.replace(
            "    auto const token = CpuGreedyArgMaxLanes8Enabled()\n"
            "                           ? ArgMaxCpuLanes8(scores)\n"
            "                           : ArgMax<float>(scores);",
            "    auto const token = ArgMax<float>(scores);")
        self.assertEqual(candidate, original)

    def test_fixed_lanes_strict_comparisons_and_process_selector(self):
        self.assertIn("values.size() - i >= 8", ARGMAX)
        self.assertIn("std::fill_n(maxima, 8, values[0])", ARGMAX)
        self.assertIn("indices[lane] < index", ARGMAX)
        self.assertIn("value > maxima[lane]", ARGMAX)
        self.assertIn("values[i] > best", ARGMAX)
        self.assertIn("static const bool enabled", SELECTOR)
        self.assertIn(f'GetEnv("{FLAG}", value)', SELECTOR)
        self.assertIn("bool value = false", SELECTOR)

    def test_standard_uses_original_runtime_controls_and_all_contexts(self):
        from run_cpu_greedy_argmax_lanes8_standard import DIRECTORY, RUNTIME, TEMPLATE, model_command
        from run_genai_allocator_workers import OLD_RUNTIME

        root = Path(__file__).parent
        template = json.loads((root / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "lanes8"):
                    command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                            root / "cpu-argmax-command-test", context, full, mode)
                    self.assertIn("--tokens 256 --repetitions 3", command[-1])
                    self.assertIn(f"--contexts {context}", command[-1])
                    self.assertIn(f"export {FLAG}={int(mode == 'lanes8')}; ", command[-1])
                    candidate = str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace")
                    self.assertIn(OLD_RUNTIME if mode == "control" else candidate, command[-1])
                    self.assertNotIn(candidate if mode == "control" else OLD_RUNTIME, command[-1])


if __name__ == "__main__":
    unittest.main()
