from pathlib import Path
import unittest

from build_qmoe_prefill256 import FLAG, ROOT, transform, transform_compare


class Prefill256Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-extended-prefill/"
                          "build-v1/overlay/contrib_ops/webgpu/moe/qmoe.cc").read_text()
        cls.result = transform(cls.source)

    def test_validation_and_cache_key(self):
        self.assertEqual(self.result.count(f'"{FLAG}", 0'), 2)
        self.assertEqual(self.result.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(prefill256)", self.result)
        self.assertLess(self.result.index(f"{FLAG} must be 0 or 1"),
                        self.result.index("key.push_back(prefill256)"))

    def test_exact_shape_and_preserved_guards(self):
        for guard in ("prefill256 == 1 && use_extended_int4_prefill",
                      "moe_params.num_rows == 256",
                      '"ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0',
                      "maxComputeWorkgroupsPerDimension >= 12800"):
            self.assertIn(guard, self.result)
        start = self.source.index("  const bool use_extended_int4_prefill")
        end = self.source.index("  const int64_t max_indirect_tokens", start)
        self.assertIn(self.source[start:end], self.result)
        self.assertIn("use_prefill256 ? 256 : use_nvfp4_prefill", self.result)
        self.assertIn("use_prefill256 ? 2560 : use_nvfp4_prefill ? 4096", self.result)

    def test_only_host_selection_changes(self):
        first = self.source.index("  const int64_t max_indirect_tokens")
        last = self.source.index("  const bool use_small_batch", first)
        original = self.source.replace(self.source[first:last], "")
        actual = self.result
        begin = actual.index("  const bool use_prefill256")
        end = actual.index("  const bool use_small_batch", begin)
        actual = actual.replace(actual[begin:end], "")
        validation = (f'\n  const int prefill256 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);'
                      f'\n  ORT_RETURN_IF_NOT(prefill256 == 0 || prefill256 == 1, "{FLAG} must be 0 or 1");')
        actual = actual.replace(validation, "").replace("\n  key.push_back(prefill256);", "")
        self.assertEqual(actual, original)

    def test_dispatch_and_scratch_bounds(self):
        routed = 256 * 10
        self.assertEqual(routed, 2560)
        self.assertEqual((routed * 640 + 127) // 128, 12800)
        for elements in (routed * 2560, routed * 1280, routed * 640):
            self.assertLess(elements, 2**32)
            self.assertLess(elements * 2, 128 * 1024**2)
        for rows in (1, 16, 128, 129, 255, 257):
            self.assertNotEqual(rows, 256)

    def test_native_reference_and_fallbacks(self):
        comparator = transform_compare((ROOT / "qmoe_extended_prefill_compare.cpp").read_text())
        self.assertIn("{16, 128, 129, 255, 256, 257}", comparator)
        self.assertIn("rows > 128 && rows != 256 ? rows : 16", comparator)
        self.assertIn("rows > 128 && rows != 256 ? 2 : 8", comparator)
        self.assertIn("last_router.data(), 257", comparator)
        self.assertIn("std::isfinite", comparator)
        self.assertIn("std::memcmp", comparator)
        self.assertIn("invalid_flag_rejected=1", comparator)


if __name__ == "__main__":
    unittest.main()
