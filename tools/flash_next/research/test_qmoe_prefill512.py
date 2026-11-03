from pathlib import Path
import unittest

from build_qmoe_prefill512 import FLAG, ROWS, transform, transform_compare
from run_qmoe_prefill256_native import verify_gate_coverage
from run_prefill256_model import verify_metadata
from test_prefill256_validation import gate
import test_qualify_qsa_score16_standard as fixtures


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1")


class Prefill512Tests(unittest.TestCase):
    def test_metadata_requires_explicit_matching_chunk(self):
        fixture = fixtures.StandardScore16Tests()
        fixture.setUp()
        fixture.raw["chunk"] = 512
        with self.assertRaises(AssertionError):
            verify_metadata(fixture.raw, fixture.guard, fixture.native, fixture.parent, True)
        verify_metadata(fixture.raw, fixture.guard, fixture.native, fixture.parent, True, chunk=512)
        for value in (256, 511, 513):
            fixture.raw["chunk"] = value
            with self.assertRaises(AssertionError):
                verify_metadata(fixture.raw, fixture.guard, fixture.native, fixture.parent, True, chunk=512)

    def test_exact_native_512_gate_coverage_rejects_missing_or_fallback(self):
        expected = {16: 464, 128: 8, 256: 8, 512: 8}
        events = [gate(f"{rows},512") for rows, count in expected.items() for _ in range(count)]
        self.assertEqual(verify_gate_coverage(events, expected), expected)
        with self.assertRaises(AssertionError):
            verify_gate_coverage(events[:-1], expected)
        with self.assertRaises(AssertionError):
            verify_gate_coverage(events + [gate("513,512")], expected)

    def test_exact_extension_and_preserved_arithmetic(self):
        source = (DIRECTORY / "overlay/contrib_ops/webgpu/moe/qmoe.cc").read_text()
        actual = transform(source)
        validation = (f'\n  const int prefill512 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);'
                      f'\n  ORT_RETURN_IF_NOT(prefill512 == 0 || prefill512 == 1, "{FLAG} must be 0 or 1");')
        self.assertEqual(actual.count(validation), 2)
        start = actual.index("  const bool use_prefill512 =")
        stop = actual.index("  const int64_t max_indirect_tokens =", start)
        eligibility = actual[start:stop]
        for condition in ("prefill512 == 1 && use_extended_int4_prefill",
                          "moe_params.num_rows == 512", "maxComputeWorkgroupsPerDimension >= 25600",
                          '"ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0'):
            self.assertIn(condition, eligibility)
        restored = (actual[:start] + actual[stop:]).replace(validation, "")
        restored = (restored.replace("\n  key.push_back(prefill512);", "")
                    .replace("use_prefill512 ? 512 : ", "").replace("use_prefill512 ? 5120 : ", ""))
        self.assertEqual(restored, source)

    def test_comparator_boundary_rows_and_chunk16_reference(self):
        source = (DIRECTORY / "native_compare.cpp").read_text()
        actual = transform_compare(source)
        self.assertEqual(ROWS, (16, 128, 256, 511, 512, 513))
        self.assertIn("{16, 128, 256, 511, 512, 513}", actual)
        self.assertIn("rows > 256 && rows != 512 ? rows : 16", actual)
        self.assertIn("rows > 256 && rows != 512 ? 2 : 8", actual)
        self.assertIn("last_router.data(), 513", actual)
        self.assertIn("retained_chunk16_reference=1 invalid_flag_rejected=1", actual)


if __name__ == "__main__":
    unittest.main()
