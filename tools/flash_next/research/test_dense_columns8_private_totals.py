from pathlib import Path
import unittest

from build_dense_columns8_private_totals import CLASS, FLAG, PROGRAM, SOURCE, transform
from build_dense_columns8_double_buffer import PROGRAM as ORIGINAL_PROGRAM
from test_hc_projection_inner_unroll import render


class DenseColumns8PrivateTotalsTests(unittest.TestCase):
    def test_actual_renderer_fp16_zeroing_order_and_single_final_store(self):
        shader = render(PROGRAM, 1, False, False, helper="", flag=FLAG,
                        class_name=CLASS, constructor_args=())
        self.assertIn("var totals = vec2<f16>(0);", shader)
        self.assertIn("totals += vec2<f16>(sum0, sum1);", shader)
        self.assertNotIn("partials[i] = f16(0)", shader)
        self.assertNotIn("] += sum", shader)
        self.assertIn("for (var word = 0u; word < 4u; word++)", shader)
        self.assertIn("sum0 += dot(x0, w0) + dot(x1, w1)", shader)
        self.assertIn("sum1 += dot(x0, w0) + dot(x1, w1)", shader)
        self.assertIn("partials[column_group * 32u + stream] = totals.x", shader)
        self.assertIn("partials[(column_group + 4u) * 32u + stream] = totals.y", shader)
        self.assertEqual(shader.count("workgroupBarrier();"), 2)
        self.assertIn("value += partials[local_idx * 32u + stream]", shader)

    def test_unique_final_shared_stores(self):
        stores = []
        for local in range(128):
            stream, group = local % 32, local // 32
            stores.extend((group * 32 + stream, (group + 4) * 32 + stream))
        self.assertEqual(sorted(stores), list(range(256)))

    def test_parent_and_fallbacks_unchanged(self):
        source = Path(str(SOURCE).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        candidate = transform(source)
        self.assertIn(ORIGINAL_PROGRAM, candidate)
        self.assertIn(f"{FLAG} must be 0 or 1", candidate)
        self.assertIn("if (columns8 && private_totals == 1)", candidate)
        self.assertNotIn("DenseColumns8WordUnrollProgram", candidate)
        self.assertNotIn("dense_swizzled_a_index", candidate)
        self.assertTrue(candidate.endswith(source[source.index("\nStatus ApplyMatMulNBits("):]))

    def test_controllers_keep_parent_and_selected_dispatch_gates(self):
        from prepare_dense_columns8_private_totals_proofs import batch_source
        for actual in (False, True):
            source = batch_source(actual)
            self.assertIn(f'FLAG = "{FLAG}"', source)
            self.assertIn('FLAGS += " ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER=1"', source)
            self.assertIn('"DenseColumns8PrivateTotals"', source)
            self.assertIn('phase == "default" and n >= 2560', source)
            self.assertNotIn("DenseColumns8WordUnroll", source)


if __name__ == "__main__":
    unittest.main()
