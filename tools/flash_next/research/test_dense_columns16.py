import ast
from pathlib import Path
import unittest

from build_dense_columns16 import CLASS, FLAG, PROGRAM, STORAGE_BYTES, transform
from prepare_dense_columns16_proofs import adapt, synthetic_runner, actual_runner
from test_hc_projection_inner_unroll import render


class Columns16Tests(unittest.TestCase):
    def test_actual_renderer_activation_reuse_and_rounding(self):
        code = render(PROGRAM, 0, False, False, "", FLAG, class_name=CLASS, constructor_args=())
        self.assertEqual(code.count("let x0 ="), 1)
        self.assertEqual(code.count("let x1 ="), 1)
        self.assertEqual(code.count("dot(x0, w0) + dot(x1, w1)"), 4)
        self.assertEqual(code.count("workgroupBarrier();"), 2)
        for j in range(4):
            self.assertIn(f"var sum{j} = f16(0)", code)
            self.assertIn(f"partials[(column_group + {j * 4}u) * 32u + stream] += sum{j}", code)
        self.assertIn("value += partials[local_idx * 32u + stream]", code)
        self.assertNotIn("subgroupMatrix", code)
        self.assertEqual(STORAGE_BYTES, 256 * 8 + 512 * 2)

    def test_mapping_all_original_streams_for_sixteen_columns(self):
        indices = [(lane % 32, lane // 32 + 4 * j) for lane in range(128) for j in range(4)]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(set(indices), {(stream, col) for stream in range(32) for col in range(16)})

    def test_default_admission_replay_and_full_reference_runners(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc").read_text()
        candidate = transform(source)
        self.assertIn(FLAG + " must be 0 or 1", candidate)
        self.assertIn("columns16_mode == 1 && dense_bitcast_route && block32_mode == 1", candidate)
        self.assertIn("columns16 ? 5", candidate)
        self.assertTrue(candidate.endswith(source[source.index("\nStatus ApplyMatMulNBits("):]))
        for runner in (synthetic_runner(), actual_runner()):
            result = adapt(runner)
            ast.parse(result)
            self.assertNotIn("DensePairedMatrixReset", result)
            self.assertIn('"DenseColumns16"', result)
            self.assertIn("source_and_evidence_sha256", result)
            self.assertIn("model_eligible=False", result)


if __name__ == "__main__":
    unittest.main()
