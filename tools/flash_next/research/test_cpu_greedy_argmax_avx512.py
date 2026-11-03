from pathlib import Path
import subprocess
import tempfile
import unittest

from build_cpu_greedy_argmax_avx512 import ARGMAX, FLAG, comparator_source, transform_source


SOURCE = Path("/home/sayanshaw/repos/onnxruntime-genai-webgpu-fp8-checkpoint/src")


class Avx512ArgMaxTests(unittest.TestCase):
    def test_actual_cpp_matches_original_and_lanes8(self):
        with tempfile.TemporaryDirectory(prefix="qwen-argmax-avx512-") as directory:
            cpp, executable = Path(directory) / "compare.cpp", Path(directory) / "compare"
            cpp.write_text(comparator_source((SOURCE / "search.h").read_text()))
            subprocess.run(["c++", "-std=c++20", "-O3", str(cpp), "-o", str(executable)], check=True)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
            self.assertIn("selected_ids_exact=1 signed_zeros_nan_inf_tested=1 input_unchanged=1", result.stdout)
            self.assertEqual(result.stdout.count("CPU_ONLY trial="), 5)
            print(result.stdout, end="")

    def test_feature_gate_and_original_sampling_route(self):
        original = (SOURCE / "search.cpp").read_text()
        source = transform_source(original)
        self.assertIn(f'GetEnv("{FLAG}", value)', source)
        self.assertIn("value && !CpuArgMaxAvx512Available()", source)
        self.assertIn("__builtin_cpu_supports(\"avx512f\")", ARGMAX)
        self.assertIn('__attribute__((target("avx512f")))', ARGMAX)
        self.assertIn("_CMP_GT_OQ", ARGMAX)
        self.assertIn(": CpuGreedyArgMaxLanes8Enabled()", source)
        self.assertIn(": ArgMax<float>(scores)", source)


if __name__ == "__main__":
    unittest.main()
