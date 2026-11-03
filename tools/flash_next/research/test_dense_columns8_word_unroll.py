from pathlib import Path
import unittest

from build_dense_columns8_word_unroll import CLASS, FLAG, PROGRAM, SOURCE, transform
from build_dense_columns8_double_buffer import PROGRAM as ORIGINAL_PROGRAM
from test_hc_projection_inner_unroll import render


class DenseColumns8WordUnrollTests(unittest.TestCase):
    def test_actual_renderer_preserves_dot_and_accumulation_order(self):
        shader = render(PROGRAM, 1, False, False, helper="", flag=FLAG,
                        class_name=CLASS, constructor_args=())
        self.assertNotIn("for (var word =", shader)
        self.assertNotIn("[word]", shader)
        self.assertEqual(shader.count("workgroupBarrier();"), 2)
        self.assertEqual(shader.count("sum0 += dot(x0, w0) + dot(x1, w1)"), 4)
        self.assertEqual(shader.count("sum1 += dot(x0, w0) + dot(x1, w1)"), 4)
        previous = 0
        for word, component in enumerate("xyzw"):
            first = shader.index(f"decode_low_nibbles(packed0.{component})", previous)
            second = shader.index(f"decode_low_nibbles(packed1.{component})", first)
            self.assertLess(first, second)
            previous = second
            self.assertIn(f"stream * 8u + {word * 2}u", shader)
            self.assertIn(f"stream * 8u + {word * 2 + 1}u", shader)
        self.assertIn("value += partials[local_idx * 32u + stream]", shader)
        self.assertEqual(shader.count("{"), shader.count("}"))

    def test_parent_selector_and_program_remain_byte_identical(self):
        source = Path(str(SOURCE).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        candidate = transform(source)
        self.assertIn(ORIGINAL_PROGRAM, candidate)
        self.assertIn('ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER', candidate)
        self.assertIn(f"{FLAG} must be 0 or 1", candidate)
        self.assertIn("if (columns8 && word_unroll == 1)", candidate)
        self.assertIn("columns8 ? (word_unroll == 1 ? 8 : 7)", candidate)
        self.assertEqual(candidate.count(".SetWorkgroupSize(128).SetDispatchGroupSize(N_ / 8);"), 2)
        self.assertTrue(candidate.endswith(source[source.index("\nStatus ApplyMatMulNBits("):]))


if __name__ == "__main__":
    unittest.main()
