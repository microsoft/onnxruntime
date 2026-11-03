import ast
import json
import unittest
from pathlib import Path

from prepare_dense_columns8_double_buffer_model import (
    invalid_source, model_source, screen_source, standard_source,
)


class DoubleBufferModelTests(unittest.TestCase):
    def test_exact_native_profile_prerequisites_and_context_ladder(self):
        for source in (invalid_source(), model_source(), standard_source(), screen_source()):
            ast.parse(source)
            self.assertNotIn("double_buffer_double_buffer", source)
            self.assertNotIn("double-buffer-double-buffer", source)
        standard = standard_source()
        self.assertIn('gpu["phase"] == "profile" and gpu["context"] == 2048', standard)
        self.assertIn('gpu["experiment_flag"] == FLAG', standard)
        self.assertIn('["DenseColumns8DoubleBuffer"] == 158', standard)
        self.assertIn('["MatMulNBitsBlock32F16"] == 1', standard)
        self.assertIn("repeated_cpu_greedy_local8_supported_all_contexts", standard)
        self.assertNotIn("context2048-repeat-screen-20261031", standard)
        self.assertIn("previous = 2048 if args.context == 8192 else 8192", standard)
        self.assertIn("repeated_columns8_double_buffer_cpu_local8_decode_gain_both_modes", standard)
        model = model_source()
        self.assertIn("columns8_double_buffer_batched_actual_and_default_supported", model)
        self.assertIn("columns8_double_buffer_batched_synthetic_and_default_supported", model)
        self.assertIn("fresh_invalid_columns8_double_buffer_selector_rejected", model)

    def test_same_cpu_runtime_and_exact_worker_placement(self):
        namespace = {"__name__": "command-test", "__file__": __file__}
        exec(compile(standard_source(), "standard", "exec"), namespace)
        root = Path(__file__).parent
        template = json.loads((root / f'{namespace["TEMPLATE"]}.guard.json').read_text())["command"]
        for mode in ("control", "columns8"):
            command = namespace["model_command"](
                template, Path("/workspace/provider.so"), root / "unused-command-proof",
                32768, True, mode)
            self.assertEqual(command[:3], ["taskset", "-c", "48-55"])
            self.assertIn("qwen38-int4-cpu-greedy-argmax/build-v1", command[-1])
            self.assertIn("export ORT_GENAI_CPU_GREEDY_ARGMAX_LANES8=1; ", command[-1])
            self.assertIn(
                f"export ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER={int(mode == 'columns8')}; ",
                command[-1])
            self.assertIn("--inherit-worker-cpu-affinity", command[-1])
        screen = screen_source()
        self.assertIn('parent_runtime_sha256=build["runtime_sha256"]', screen)
        self.assertIn("original_baseline_provider_sha256=", screen)


if __name__ == "__main__":
    unittest.main()
