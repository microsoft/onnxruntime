import unittest

from prepare_dense_long_k_native_v2 import ROOT, build_source, controller_source, native_source


class RankAwareNativeTests(unittest.TestCase):
    def test_only_rank_contract_changes_with_prefix_dimensions_and_positive_shapes_checked(self):
        source = (ROOT / "dense_long_k_fixed_native.cpp").read_text()
        result = native_source(source)
        self.assertIn("shape.size() < 2 || output_shape.size() != shape.size()", result)
        self.assertIn("!std::equal(shape.begin(), shape.end() - 1, output_shape.begin())", result)
        self.assertIn("std::any_of(shape.begin(), shape.end()", result)
        self.assertIn("std::memcmp(&expected[i], &observed[i], sizeof(expected[i]))", result)
        self.assertIn("info.GetShape() != output_shape || info.GetElementCount() != expected.size()", result)
        self.assertIn("private_value() != initial_private", result)
        builder = build_source((ROOT / "build_dense_long_k_native.py").read_text())
        controller = controller_source((ROOT / "run_dense_long_k_native_v1.py").read_text())
        self.assertIn("native-full-v2", builder)
        self.assertIn("dense_long_k_fixed_native_v2.cpp", builder)
        self.assertIn("native-proof-v2", controller)
        self.assertIn('VARIANT = "dense-tile-parallel3-long-k-native-v2"', controller)
        compile(builder, "rank-aware-builder", "exec")
        compile(controller, "rank-aware-controller", "exec")


if __name__ == "__main__":
    unittest.main()
