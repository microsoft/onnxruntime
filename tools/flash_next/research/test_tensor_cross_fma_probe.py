from pathlib import Path
import unittest

from build_tensor_cross_fma_probe import CROSS_DOT, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/matmul_nbits.cc")


class TensorCrossFmaProbeTests(unittest.TestCase):
    def test_cross_pair_mapping_uses_every_term_once(self):
        products = []
        increments = []
        for row in range(16):
            if row % 4 < 2:
                products.append((row, row))
                increments.append((row, row + 2))
        for start in (0, 4, 8, 12):
            self.assertEqual([k for row, k in products + increments if start <= row < start + 4],
                             [start, start + 1, start + 2, start + 3])

    def test_probe_uses_two_native_half_accumulations_and_original_fixture_guard(self):
        actual = transform(SOURCE.read_text())
        self.assertIn('Program{"TensorCrossFmaProbeF16"}', actual)
        self.assertIn(CROSS_DOT, actual)
        self.assertEqual(actual.count("result = subgroupMatrixMultiplyAccumulate("), 2)
        self.assertIn("row == col && row % 4u < 2u", actual)
        self.assertIn("col == row + 2u && row % 4u < 2u", actual)
        self.assertIn("Tensor dot4 probe only supports the isolated FP16", actual)
        self.assertIn("subgroupMatrixStore<row_major>(&products", actual)

    def test_materialized_variant_forces_half_boundary_before_second_accumulation(self):
        actual = transform(SOURCE.read_text(), materialize=True)
        self.assertIn("subgroupMatrixLoad<subgroup_matrix_result<f16, 16, 16>, row_major>", actual)
        self.assertIn("subgroupMatrixMultiplyAccumulate(shifted_left, right, rounded_result)", actual)
        self.assertEqual(actual.count("subgroupMatrixStore<row_major>(&products"), 2)
        self.assertLess(actual.index("subgroupMatrixStore<row_major>(&products"),
                        actual.index("let shifted_left"))
        self.assertLess(actual.index("let rounded_result"),
                        actual.index("subgroupMatrixMultiplyAccumulate(shifted_left"))

    def test_fp32_variant_has_observable_half_boundaries(self):
        actual = transform(SOURCE.read_text(), fp32_stages=True)
        self.assertIn('Program{"TensorCrossFmaF32Probe"}', actual)
        self.assertIn("subgroup_matrix_result<f32, 16, 16>", actual)
        self.assertIn("unpack2x16float(pack2x16float(vec2<f32>(scratch_f32[i], 0)))", actual)
        self.assertIn("products[i] = f16(scratch_f32[i])", actual)
        self.assertEqual(actual.count("subgroupMatrixStore<row_major>(&scratch_f32"), 2)
        self.assertIn("subgroupMatrixComponentType::F32".lower(),
                      actual.lower())


if __name__ == "__main__":
    unittest.main()
