from pathlib import Path
import unittest

from build_tensor_f32_products_probe import transform
from build_tensor_product_probe import transform as products
from run_tensor_dot4_probe import classify_result


class TensorF32ProductsTests(unittest.TestCase):
    def test_only_independent_product_precision_and_dot_precision_change(self):
        original = products(Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/"
                                 "matmul_nbits.cc").read_text())
        actual = transform(Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/"
                                "matmul_nbits.cc").read_text())
        replacements = (
            ('"TensorProductProbeF16"', '"TensorProductF32ProbeF16"'),
            ("products: array<f16, 256>", "products: array<f32, 256>"),
            ("let p = vec4<f16>(products[", "let p = vec4<f32>(products["),
            ("return dot(vec4<f16>(1), p);", "return f16(dot(vec4<f32>(1), p));"),
            ("subgroup_matrix_result<f16, 16, 16>", "subgroup_matrix_result<f32, 16, 16>"),
            ("{{f16, f16, 16, 16, 16, 32, true}}",
             "{{f16, wgpu::SubgroupMatrixComponentType::F32, 16, 16, 16, 32, true}}"),
            ("Tensor dot4 probe requires native FP16 16x16x16 matrices",
             "Tensor product probe requires native FP16 inputs and FP32-result 16x16x16 matrices"),
        )
        for old, new in reversed(replacements):
            self.assertIn(new, actual)
            actual = actual.replace(new, old, 1)
        self.assertEqual(actual, original)

    def test_infrastructure_failure_cannot_be_numerical_rejection(self):
        with self.assertRaises(AssertionError):
            classify_result({"exit_code": 1, "status": "failed"}, "FAIL shader compilation failed", 1)
        status, reason = classify_result({"exit_code": 1, "status": "failed"},
                                         "FAIL finite/bitwise mismatch element=4", 1)
        self.assertEqual(status, "native_tensor_dot4_not_bitwise_compatible")
        self.assertIn("mismatch", reason)


if __name__ == "__main__":
    unittest.main()
