from collections import Counter
from pathlib import Path
import unittest

from build_dense_columns8_swizzle import CLASS, FLAG, PROGRAM, SOURCE, transform
from build_dense_columns8_double_buffer import PROGRAM as ORIGINAL_PROGRAM
from test_hc_projection_inner_unroll import render


def address(index):
    stream = index // 8
    return stream * 8 + ((index % 8) ^ ((stream >> 1) & 7))


class DenseColumns8SwizzleTests(unittest.TestCase):
    def test_bijective_storage_and_exact_loader_roundtrip(self):
        self.assertEqual(sorted(address(i) for i in range(256)), list(range(256)))
        storage = [None] * 512
        for tile in range(9):
            offset = tile % 2 * 256
            for i in range(256):
                storage[offset + address(i)] = (tile, i)
            for stream in range(32):
                for word in range(8):
                    i = stream * 8 + word
                    self.assertEqual(storage[offset + address(i)], (tile, i))

    def test_modeled_nvidia_bank_multiplicity_and_staging(self):
        for word in range(8):
            original = Counter((stream * 8 + word) * 2 % 32 for stream in range(32))
            swizzled = Counter(address(stream * 8 + word) * 2 % 32 for stream in range(32))
            self.assertEqual(max(original.values()), 16)
            self.assertEqual(max(swizzled.values()), 2)
        for first in range(0, 256, 32):
            writes = Counter(address(i) * 2 % 32 for i in range(first, first + 32))
            self.assertEqual(max(writes.values()), 2)

    def test_actual_renderer_arithmetic_and_barriers_unchanged(self):
        shader = render(PROGRAM, 1, False, False, helper="", flag=FLAG,
                        class_name=CLASS, constructor_args=())
        self.assertEqual(shader.count("workgroupBarrier();"), 2)
        self.assertIn("for (var word = 0u; word < 4u; word++)", shader)
        self.assertIn("dense_swizzled_a_index(stream * 8u + word * 2u)", shader)
        self.assertIn("dense_swizzled_a_index(stream * 8u + word * 2u + 1u)", shader)
        self.assertIn("sum0 += dot(x0, w0) + dot(x1, w1)", shader)
        self.assertIn("sum1 += dot(x0, w0) + dot(x1, w1)", shader)
        self.assertIn("value += partials[local_idx * 32u + stream]", shader)
        self.assertNotIn("DenseColumns8WordUnroll", PROGRAM)

    def test_original_parent_default_and_hardware_guard(self):
        source = Path(str(SOURCE).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        candidate = transform(source)
        self.assertIn(ORIGINAL_PROGRAM, candidate)
        self.assertIn(f"{FLAG} must be 0 or 1", candidate)
        self.assertIn('context.AdapterInfo().vendor == std::string_view{"nvidia"}', candidate)
        self.assertIn("columns8 ? (use_swizzle ? 8 : 7)", candidate)
        self.assertNotIn("DenseColumns8WordUnrollProgram", candidate)
        self.assertTrue(candidate.endswith(source[source.index("\nStatus ApplyMatMulNBits("):]))

    def test_both_generated_controllers_require_selected_and_parent_dispatches(self):
        from prepare_dense_columns8_swizzle_proofs import batch_source
        for actual in (False, True):
            source = batch_source(actual)
            self.assertIn(f'FLAG = "{FLAG}"', source)
            self.assertIn('FLAGS += " ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER=1"', source)
            self.assertIn('"DenseColumns8Swizzle"', source)
            self.assertIn('"DenseColumns8DoubleBuffer"', source)
            self.assertIn('phase == "default" and n >= 2560', source)
            self.assertNotIn("DenseColumns8WordUnroll", source)


if __name__ == "__main__":
    unittest.main()
