import ast
import unittest
from pathlib import Path

from prepare_dense_columns8 import builder_source
from test_hc_projection_inner_unroll import render


class Columns8Tests(unittest.TestCase):
    def test_actual_generated_shader_and_mapping(self):
        namespace = {"__name__": "columns8-test", "__file__": __file__}
        exec(compile(builder_source(), "builder", "exec"), namespace)
        code = render(namespace["PROGRAM"], 0, False, False, "", namespace["FLAG"],
                      class_name=namespace["CLASS"], constructor_args=())
        self.assertEqual(code.count("dot(x0, w0) + dot(x1, w1)"), 2)
        self.assertIn("partials: array<f16, 256>", code)
        self.assertIn("local_idx < 8u", code)
        self.assertIn("workgroup_idx * 8u", code)
        self.assertNotIn("sum2", code)
        self.assertEqual(namespace["STORAGE_BYTES"], 256 * 8 + 256 * 2)
        source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc").read_text()
        candidate = namespace["transform"](source)
        self.assertIn("N_ >= 2560 && N_ % 8 == 0", candidate)
        self.assertIn("columns8 ? 6", candidate)
        self.assertIn("TypeAndRank, 16", candidate)
        indices = [(lane % 32, lane // 32 + 4 * j) for lane in range(128) for j in range(2)]
        self.assertEqual(set(indices), {(stream, col) for stream in range(32) for col in range(8)})

    def test_batched_driver_keeps_original_default_and_candidate_distinct(self):
        path = Path(__file__).with_name("run_dense_columns8_batch.py")
        source = path.read_text()
        ast.parse(source)
        self.assertIn('("capture", ORIGINAL, "capture", 0)', source)
        self.assertIn('("default", provider, "compare", 0)', source)
        self.assertIn('("candidate", provider, "compare", 1)', source)
        self.assertIn('"; ".join(commands)', source)
        self.assertIn("model_eligible=False", source)
        self.assertIn("natural_model_timing_performed=False", source)


if __name__ == "__main__":
    unittest.main()
