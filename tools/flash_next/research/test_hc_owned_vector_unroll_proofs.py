import ast
import collections
import json
from pathlib import Path
import unittest

from build_cpu_greedy_argmax_lanes8 import FLAG as CPU_FLAG
from build_hc_owned_vector_unroll import FLAG, NAME
from prepare_hc_owned_vector_unroll_proofs import ROOT, benchmark_source, controller_source
from run_qmoe_prefill_rows4_standard import TEMPLATE


class OwnedVectorUnrollProofTests(unittest.TestCase):
    def test_only_benchmark_flag_recording_changes(self):
        self.assertEqual(benchmark_source().replace(f'                "{FLAG}",\n', ""),
                         (ROOT / "benchmark_webgpu_int4.py").read_text())

    def test_native_and_full_model_requirements(self):
        for kind in ("native", "model", "cpu_local8_standard", "cpu_local8_screen"):
            ast.parse(controller_source(kind))
        namespace = {"__name__": "hc-unroll-model-test", "__file__": __file__}
        exec(compile(controller_source("model"), "hc-unroll-model", "exec"), namespace)
        counts = collections.Counter({NAME: 94, "MatMulNBits": 1,
            "MatMulNBitsBlock32F16": 159, "DynamicSparseAttentionQueryCache": 12})
        namespace["verify_profile"](counts, "candidate")
        with self.assertRaises(AssertionError):
            namespace["verify_profile"](collections.Counter({NAME: 94}), "candidate")

    def test_all_context_cpu_worker_and_runtime_policy(self):
        namespace = {"__name__": "hc-unroll-cpu-test", "__file__": __file__}
        exec(compile(controller_source("cpu_local8_standard"), "hc-unroll-cpu", "exec"), namespace)
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "candidate"):
                    command = namespace["model_command"](template, Path("/home/sayanshaw/repos/provider.so"),
                        ROOT / "test-prefix", context, full, mode)
                    self.assertEqual(command[:3], ["taskset", "-c", "48-55"])
                    self.assertIn(f"export {FLAG}={int(mode == 'candidate')}", command[-1])
                    self.assertIn(f"{CPU_FLAG}=1", command[-1])
                    self.assertIn("--inherit-worker-cpu-affinity", command[-1])
                    self.assertIn("benchmark_webgpu_int4_hc_owned_vector_unroll.py", command[-1])
                    self.assertIn(f"--contexts {context}", command[-1])
                    self.assertEqual("--webgpu-validation-mode full" in command[-1], full)


if __name__ == "__main__":
    unittest.main()
