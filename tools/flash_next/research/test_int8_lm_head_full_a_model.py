import ast
import collections
import json
from pathlib import Path
import unittest

from build_int8_lm_head_full_a import FLAG, NAME
from build_cpu_greedy_argmax_lanes8 import FLAG as CPU_FLAG
from prepare_int8_lm_head_full_a_model import (
    ROOT, benchmark_source, cpu_source, model_source, screen_source,
)
from run_qmoe_prefill_rows4_standard import TEMPLATE


class Int8ModelTests(unittest.TestCase):
    def test_benchmark_only_records_new_flag(self):
        self.assertEqual(benchmark_source().replace(f'                "{FLAG}",\n', ""),
                         (ROOT / "benchmark_webgpu_int4.py").read_text())

    def test_generated_controllers_and_required_coverage(self):
        for source in (model_source(), cpu_source(), screen_source()):
            ast.parse(source)
            self.assertNotIn("hc_wide_rms_broadcast", source)
        namespace = {"__name__": "int8-model-test", "__file__": __file__}
        exec(compile(model_source(), "int8-model", "exec"), namespace)
        counts = collections.Counter({NAME: 1, "HCPostNormProjectionF16": 94,
            "MatMulNBitsBlock32F16": 159, "DynamicSparseAttentionQueryCache": 12})
        namespace["verify_profile"](counts, "candidate")
        with self.assertRaises(AssertionError):
            namespace["verify_profile"](collections.Counter({NAME: 1}), "candidate")
        self.assertIn("15906816", model_source())
        self.assertIn("int8-lm-head-full-a-v4-native-outcome.json", model_source())
        self.assertIn('ROOT / "prepare_int8_lm_head_full_a_model.py"', model_source())
        self.assertIn('["Int8LmHeadFullAF16"] == 1', cpu_source())
        self.assertIn('["HCPostNormProjectionF16"] == 94', cpu_source())

    def test_all_contexts_same_cpu_runtime_and_worker_placement(self):
        namespace = {"__name__": "int8-cpu-test", "__file__": __file__}
        exec(compile(cpu_source(), "int8-cpu", "exec"), namespace)
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (True, False):
                for mode in ("control", "candidate"):
                    command = namespace["model_command"](template, Path("/home/sayanshaw/repos/provider.so"),
                        ROOT / "test-prefix", context, full, mode)
                    self.assertIn(f"export {FLAG}={int(mode == 'candidate')}", command[-1])
                    self.assertIn(f"{CPU_FLAG}=1", command[-1])
                    self.assertEqual(command[:3], ["taskset", "-c", "48-55"])
                    self.assertIn("--inherit-worker-cpu-affinity", command[-1])
                    self.assertIn("benchmark_webgpu_int4_int8_head.py", command[-1])
                    self.assertIn(f"--contexts {context}", command[-1])
                    self.assertEqual("--webgpu-validation-mode full" in command[-1], full)


if __name__ == "__main__":
    unittest.main()
