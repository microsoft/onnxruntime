from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

from build_tensor_dot4_probe import FLAG, transform
from run_tensor_dot4_probe import classify_result
from build_tensor_product_probe import transform as transform_products


class TensorDot4ProbeTests(unittest.TestCase):
    def test_probe_is_explicitly_scoped_and_preserves_original_routes(self):
        original = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/matmul_nbits.cc").read_text()
        actual = transform(original)
        self.assertTrue(f"{FLAG} must be 0 or 1" in actual)
        self.assertTrue("K_ == 2560 && N_ == 2560" in actual)
        self.assertTrue("!context.EnableMatmulFp32Accumulation()" in actual)
        self.assertTrue("return ApplySubgroupMatrixMatMulNBits(" in actual)
        marker = "  const bool dense_bitcast_route ="
        start, stop = original.index(marker), original.index("\n}\n", original.index(marker))
        self.assertTrue(original[start:stop] in actual)
        f32 = transform(original, fp32_result=True)
        self.assertTrue("{{f16, wgpu::SubgroupMatrixComponentType::F32, 16, 16, 16, 32, true}}" in f32)
        self.assertTrue(original[start:stop] in f32)

    def test_numerical_failures_are_rejected_not_confused_with_guard_failure(self):
        status, reason = classify_result({"status": "failed", "exit_code": 1},
                                         "FAIL fixture finite/bitwise mismatch round=0", 1)
        self.assertEqual(status, "native_tensor_dot4_not_bitwise_compatible")
        self.assertIn("finite/bitwise mismatch", reason)
        for guard, log in (({"status": "guard_failed", "exit_code": 1}, reason),
                           ({"status": "failed", "exit_code": 1}, "FAIL missing native matrix feature"),
                           ({"status": "completed", "exit_code": 0}, "PASS partial")):
            with self.assertRaises(AssertionError):
                classify_result(guard, log, guard["exit_code"])

    def test_product_probe_materializes_independent_half_products(self):
        original = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/matmul_nbits.cc").read_text()
        actual = transform_products(original)
        self.assertIn("row == col", actual)
        self.assertIn("subgroupMatrixStore<row_major>(&products", actual)
        self.assertIn("subgroup_matrix_left<f16, 16, 16>, row_major>", actual)
        self.assertIn("subgroup_matrix_right<f16, 16, 16>, col_major>", actual)
        self.assertIn("return dot(vec4<f16>(1), p);", actual)
        self.assertIn("program.SetWorkgroupSize(32).SetDispatchGroupSize(160)", actual)
        marker = "  const bool dense_bitcast_route ="
        self.assertIn(original[original.index(marker):original.index("\n}\n", original.index(marker))], actual)

    def test_half_product_rounding_explains_first_rejected_element(self):
        a = np.array([0.0087432861328125, 0.0174407958984375,
                      0.02606201171875, 0.0345458984375], dtype=np.float16)
        b = np.array([0.0028820037841796875, 0.020172119140625,
                      0.020172119140625, 0.020172119140625], dtype=np.float16)
        products = a.astype(np.float32) * b.astype(np.float32)
        self.assertEqual(int(np.float16(products.sum()).view(np.uint16)), 5773)
        half = products.astype(np.float16)
        tree = np.float16(np.float16(half[0] + half[1]) + np.float16(half[2] + half[3]))
        self.assertEqual(int(tree.view(np.uint16)), 5774)

    def test_prepared_fixtures_have_exactly_one_four_product_group(self):
        root = Path("/home/sayanshaw/repos/qwen38-int4-tensor-dot4-probe/fixtures-v1")
        for offset in (0, 4, 12):
            model = onnx.load(root / f"dot4_offset{offset}.onnx")
            tensors = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
            packed = tensors["weights"]
            unpacked = np.empty((2560, 2560), dtype=np.uint8)
            unpacked[:, 0::2] = (packed.reshape(2560, -1) & 15)
            unpacked[:, 1::2] = (packed.reshape(2560, -1) >> 4)
            allowed = np.zeros(2560, dtype=bool)
            allowed[offset:offset + 4] = True
            self.assertTrue(np.all(unpacked[:, ~allowed] == 8))
            self.assertTrue(np.any(unpacked[:, allowed] != 8))
            self.assertEqual(tensors["scales"].dtype, np.float16)
            self.assertEqual(model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value, 2560)


if __name__ == "__main__":
    unittest.main()
