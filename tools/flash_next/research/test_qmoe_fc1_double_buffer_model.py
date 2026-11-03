import ast
from collections import Counter
import json
import unittest

from build_qmoe_fc1_double_buffer import FLAG
from prepare_qmoe_fc1_double_buffer_model import ROOT, invalid_source, model_source, screen_source, standard_source


class FC1ModelTests(unittest.TestCase):
    def test_independent_native_gate_and_real_dispatch_counts(self):
        source = model_source()
        ast.parse(source)
        self.assertIn("qmoe_fc1_separate_original_outputs_supported", source)
        self.assertIn('native_proof["fixtures"] == 23', source)
        self.assertIn('native_proof["comparisons_per_complete_phase"] == 9748480', source)
        self.assertIn('row["instances"] == 11 and row["hits"] > 0', source)
        namespace = {"__name__": "model-test", "__file__": str(ROOT / "run_qmoe_fc1_double_buffer_model.py")}
        exec(compile(source, "fc1-model", "exec"), namespace)
        counts = Counter(QmoeFC1SwiGLUDoubleBufferF16=48, QmoeFC2FinalMixBitcastF16=48,
                         HCPostNormProjectionF16=94, MatMulNBitsBlock32F16=159,
                         DynamicSparseAttentionQueryCache=12)
        namespace["verify_profile"](counts, "candidate")
        counts["QmoeFC1SwiGLUBitcastF16"] = 1
        with self.assertRaises(AssertionError):
            namespace["verify_profile"](counts, "candidate")

    def test_identical_cpu_runtime_worker_policy_and_context_gates(self):
        source = standard_source()
        ast.parse(source)
        namespace = {"__name__": "standard-test", "__file__": str(ROOT / "run_qmoe_fc1_double_buffer_cpu_local8_standard.py")}
        exec(compile(source, "fc1-standard", "exec"), namespace)
        from run_qmoe_prefill_rows4_standard import DIRECTORY, TEMPLATE
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "candidate"):
                    command = namespace["model_command"](template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                                        ROOT / "fc1-command-test", context, full, mode)
                    self.assertEqual(command[:3], ["taskset", "-c", "48-55"])
                    self.assertIn("export ORT_GENAI_CPU_GREEDY_ARGMAX_LANES8=1; ", command[-1])
                    self.assertIn(f"export {FLAG}={int(mode == 'candidate')}; ", command[-1])
                    self.assertIn("--inherit-worker-cpu-affinity", command[-1])
                    self.assertIn(f"--contexts {context}", command[-1])
        self.assertIn("qmoe-fc1-double-buffer-v1-candidate-gpu7-full-profile2048", source)
        self.assertIn('previous = 2048 if args.context == 8192 else 8192', source)
        self.assertIn("repeat-screen", source)
        ast.parse(screen_source())
        self.assertNotIn("hc-narrow-rms-broadcast", source)

    def test_fresh_invalid_selector_inherits_correct_fusion_flags(self):
        source = invalid_source()
        ast.parse(source)
        self.assertIn("qmoe_fc1_separate_original_outputs_supported", source)
        self.assertIn("fresh_invalid_qmoe_fc1_double_buffer_selector_rejected", source)
        self.assertIn("if env {FLAGS} {FLAG}=2", source)
        self.assertIn("from run_qmoe_fc1_double_buffer_native import", source)
        self.assertNotIn("hc-narrow-rms-broadcast", source)
        self.assertNotIn("hc-post-norm-projection", source)
        self.assertIn('if len(fixtures) != 23:', source)
        self.assertEqual(ast.dump(ast.parse(source)),
                         ast.dump(ast.parse((ROOT / "run_qmoe_fc1_double_buffer_invalid.py").read_text())))


if __name__ == "__main__":
    unittest.main()
