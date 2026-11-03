import ast
import json
from pathlib import Path
import unittest

from prepare_dense_columns8_cpu_local8 import standard_source, screen_source


class CombinedCommandTests(unittest.TestCase):
    def test_both_modes_use_qualified_cpu_runtime_mask_and_gpu_toggle(self):
        source = standard_source()
        ast.parse(source)
        namespace = {"__name__": "combined-test", "__file__": __file__}
        exec(compile(source, "standard", "exec"), namespace)
        root = Path(__file__).parent
        template = json.loads((root / (namespace["TEMPLATE"] + ".guard.json")).read_text())["command"]
        provider = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8/build-v1/libonnxruntime_providers_webgpu.so")
        for mode in ("control", "columns8"):
            command = namespace["model_command"](template, provider, root / "combined-test", 2048, True, mode)
            self.assertIn("export ORT_GENAI_CPU_GREEDY_ARGMAX_LANES8=1;", command[-1])
            self.assertIn(f"export ORT_WEBGPU_DENSE_COLUMNS8={int(mode == 'columns8')};", command[-1])
            self.assertIn("/workspace/qwen38-int4-cpu-greedy-argmax/build-v1", command[-1])
            self.assertIn("--inherit-worker-cpu-affinity", command[-1])
            self.assertEqual(command[:3], ["taskset", "-c", "48-55"])
        self.assertIn('gpu["status"] == "repeated_columns8_decode_gain_both_modes"', source)
        self.assertIn('cpu_all["status"] == "repeated_cpu_greedy_local8_supported_all_contexts"', source)

    def test_screen_compares_same_runtime_and_distinct_providers_all_contexts(self):
        source = screen_source()
        ast.parse(source)
        self.assertIn('expected_runtime = build["runtime_sha256"]', source)
        self.assertIn("original_baseline_provider_sha256", source)
        self.assertIn("fresh_reuses_initial_ending_controls=args.repeat", source)
        self.assertIn("choices=(2048, 8192, 32768)", source)
        self.assertIn("compare_samples", source)
        self.assertIn('assert proof["cpu_argmax_flag_value"] == 1', source)


if __name__ == "__main__":
    unittest.main()
