import ast
import collections
import json
from pathlib import Path
import unittest

from prepare_hc_narrow_rms_broadcast_model import model_source, screen_source, standard_source


class HCModelTests(unittest.TestCase):
    def test_native_gate_and_actual_kernel_coverage(self):
        for source in (model_source(), standard_source(), screen_source()):
            ast.parse(source)
        source = model_source()
        self.assertNotIn("synthetic_path", source)
        self.assertNotIn("columns8_double_buffer_batched", source)
        self.assertIn("hc_separate_original_outputs_supported", source)
        self.assertIn('comparisons_per_complete_phase"] == 607143936', source)
        namespace = {"__name__": "profile-test", "__file__": __file__}
        exec(compile(source, "model", "exec"), namespace)
        counts = collections.Counter(HCPostNormProjectionNarrowRmsBroadcastF16=94,
                                     MatMulNBitsBlock32F16=159,
                                     DynamicSparseAttentionQueryCache=12)
        namespace["verify_profile"](counts, "candidate")
        counts["HCPostNormProjectionF16"] = 1
        with self.assertRaises(AssertionError):
            namespace["verify_profile"](counts, "candidate")

    def test_same_runtime_worker_policy_and_context_gates(self):
        source = standard_source()
        self.assertIn("hc-narrow-rms-broadcast-v3-candidate-gpu7-full-profile2048", source)
        self.assertIn("repeated_cpu_greedy_local8_supported_all_contexts", source)
        self.assertIn("previous = 2048 if args.context == 8192 else 8192", source)
        namespace = {"__name__": "command-test", "__file__": __file__}
        exec(compile(source, "standard", "exec"), namespace)
        root = Path(__file__).parent
        template = json.loads((root / f'{namespace["TEMPLATE"]}.guard.json').read_text())["command"]
        for mode in ("control", "candidate"):
            command = namespace["model_command"](template, Path("/workspace/provider.so"),
                                                 root / "unused-command-test", 32768, True, mode)
            self.assertEqual(command[:3], ["taskset", "-c", "48-55"])
            self.assertIn("export ORT_GENAI_CPU_GREEDY_ARGMAX_LANES8=1;", command[-1])
            self.assertIn("qwen38-int4-cpu-greedy-argmax/build-v1", command[-1])
            self.assertIn(f"export ORT_WEBGPU_HC_NARROW_RMS_BROADCAST={int(mode == 'candidate')};", command[-1])
            self.assertIn("--inherit-worker-cpu-affinity", command[-1])


if __name__ == "__main__":
    unittest.main()
