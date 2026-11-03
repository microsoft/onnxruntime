from pathlib import Path
import re
import unittest

from build_dense_static_shape import DIMENSIONS, FLAG, static_template, transform
from build_dense_double_buffer import transform_compare


COMPONENT = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1")


class DenseStaticShapeTests(unittest.TestCase):
    def test_only_integer_uniform_names_and_template_name_change(self):
        source = (COMPONENT.parent / "block32-template-v1/dense_block32_template.inc").read_text()
        actual = static_template(source)
        restored = actual.replace("ApplyDenseStaticShapeTemplate", "ApplyDenseBlock32Template")
        for name in DIMENSIONS:
            restored = restored.replace(f"dense_static_{name}", f"uniforms.{name}")
        self.assertEqual(restored, source)
        self.assertNotRegex(actual, r"\buniforms\.(N|K|K_of_a|K_of_b|blocks_per_col)\b")

    def test_original_program_and_admission_are_preserved(self):
        source = (COMPONENT / "matmul_nbits.cc").read_text()
        actual = transform(source)
        start = source.index("class DenseBlock32Program")
        stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
        self.assertIn(source[start:stop], actual)
        route = source.split("const bool dense_bitcast_route =", 1)[1].split("auto compute =", 1)[0]
        self.assertIn(route, actual)
        self.assertIn("block32_mode == 1 && static_shape_mode == 1", actual)
        self.assertIn("static_shape_mode == 1 ? 4 : 3", actual)
        self.assertIn("tile_size_k_vec, broadcast_a, acc_f32, N, K, K / components_a, K_of_b, n_blocks_per_col);", actual)
        self.assertIn(f'"{FLAG}", 0', actual)
        for name in DIMENSIONS:
            self.assertIn(f"const dense_static_{name}: u32 = ", actual)
        self.assertEqual(len(re.findall("MatMulNBitsStaticShapeF16", actual)), 1)

    def test_native_fp32_comparator_uses_candidate_flag_and_keeps_old_default(self):
        source = (COMPONENT / "native_compare.cpp").read_text().replace("ORT_WEBGPU_DENSE_BLOCK32", FLAG)
        actual = transform_compare(source, flag=FLAG)
        self.assertIn('{"enableMatmulFp32Accumulation", fp32}', actual)
        with self.assertRaises(AssertionError):
            transform_compare(source)


if __name__ == "__main__":
    unittest.main()
