from pathlib import Path
import unittest

from gpu_timestamp_ns_transform import precise_duration_us, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-static-node-replay/build-v8/overlay/core/providers/webgpu/webgpu_context.cc")


class GpuTimestampPrecisionTests(unittest.TestCase):
    def test_only_additive_metadata_changes_the_frozen_actual_context(self):
        original = SOURCE.read_text()
        actual = transform(original)
        self.assertEqual(actual.replace(
            '            {"gpu_duration_ns", std::to_string(end_time - start_time)},\n', ""), original)
        self.assertIn("std::round((end_time - start_time) / 1000.0)", actual)
        self.assertEqual(actual.count('"gpu_duration_ns"'), 1)

    def test_nanoseconds_are_explicit_positive_and_consistent_with_legacy_rounding(self):
        self.assertEqual(precise_duration_us({"dur": 13, "args": {"gpu_duration_ns": "12741"}}), 12.741)
        self.assertEqual(precise_duration_us({"dur": 1, "args": {"gpu_duration_ns": "500"}}), 0.5)
        self.assertEqual(precise_duration_us({"dur": 0, "args": {"gpu_duration_ns": "499"}}), 0.499)
        for raw in ("0", "-1", "nan", "1.5", "", str(2**64), "１２３", None):
            with self.assertRaises(ValueError):
                precise_duration_us({"dur": 13, "args": {"gpu_duration_ns": raw}})
        for event in ({}, {"args": None}, {"args": []},
                      {"dur": 12, "args": {"gpu_duration_ns": "12741"}},
                      {"dur": True, "args": {"gpu_duration_ns": "1000"}}):
            with self.assertRaises(ValueError):
                precise_duration_us(event)


if __name__ == "__main__":
    unittest.main()
