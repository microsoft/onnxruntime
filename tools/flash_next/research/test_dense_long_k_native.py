from pathlib import Path
import unittest

from run_dense_long_k_native_v1 import NAME, ROOT, shape, verify_profile, verify_replay


class DenseLongKNativeTests(unittest.TestCase):
    def test_source_preserves_type_info_owners_and_fixed_private_flag_and_every_byte(self):
        source = (ROOT / "dense_long_k_fixed_native.cpp").read_text()
        self.assertIn("const auto input_type = session.GetInputTypeInfo(0)", source)
        self.assertIn("const auto input_info = input_type.GetTensorTypeAndShapeInfo()", source)
        self.assertNotIn('setenv("ORT_WEBGPU_DENSE_TILE_PARALLEL3_LONG_K"', source)
        self.assertIn("private_value() != initial_private", source)
        self.assertIn("std::memcmp(&expected[i], &observed[i], sizeof(expected[i]))", source)
        self.assertIn("Truncated original reference", source)
        self.assertIn("Trailing original reference bytes", source)
        self.assertIn("expected.size() * 128", source)

    def test_profile_and_replay_gates_demand_real_selected_and_fallback_coverage(self):
        events = [{"cat": "Kernel", "name": "node&" + NAME, "pid": -1, "tid": -1,
                   "ph": "X", "dur": 16, "args": {"cache_key": ":384,1,1:"}} for _ in range(128)]
        verify_profile(events, True, 128)
        with self.assertRaises(RuntimeError):
            verify_profile(events, False, 128)
        rows = [{"operator": "MatMulNBitsExact", "requests": 128, "hits": 126,
                 "binding_hits": 126, "captures": 2} for _ in range(27)]
        verify_replay(rows, False, 128)
        verify_replay([], True, 128)
        rows[0]["hits"] = 0
        with self.assertRaises(RuntimeError):
            verify_replay(rows, False, 128)
        self.assertEqual(shape(Path("matmul_k6144_n2560_rows16.onnx")), (6144, 2560, 16))


if __name__ == "__main__":
    unittest.main()
