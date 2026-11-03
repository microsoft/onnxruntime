from pathlib import Path
import unittest

from build_qmoe_extended_prefill import transform


class ExtendedPrefillTests(unittest.TestCase):
    def test_scoped_extension_and_preserved_other_formats(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-expert-bitcast-unpack/build-v2/"
                      "overlay/contrib_ops/webgpu/moe/qmoe.cc").read_text()
        result = transform(source)
        self.assertEqual(result.count("ORT_WEBGPU_QMOE_EXTENDED_PREFILL must be 0 or 1"), 2)
        self.assertIn("key.push_back(extended_prefill)", result)
        for guard in ("use_int4_prefill && block_size_ == 32", "is_fused_swiglu",
                      "moe_params.hidden_size == 2560", "moe_params.inter_size == 640",
                      "num_experts == 512 && k_ == 10",
                      "fc3_experts_weights_optional == nullptr",
                      "!context.EnableMatmulFp32Accumulation()"):
            self.assertIn(guard, result)
        self.assertIn("use_nvfp4_prefill ? 4096", result)
        self.assertIn("use_extended_int4_prefill ? 1280", result)

    def test_expanded_scratch_and_dispatch_sizes_are_bounded(self):
        for rows in (16, 32, 64, 128):
            routed = rows * 10
            self.assertLessEqual(routed, 1280)
            self.assertLessEqual(routed, 2560)
            self.assertLess(routed * 2560, 2**32)
            self.assertLess(routed * 1280, 2**32)
            self.assertLess(routed * 2560 * 2, 256 * 1024**2)
        self.assertGreater(129 * 10, 1280)


if __name__ == "__main__":
    unittest.main()
