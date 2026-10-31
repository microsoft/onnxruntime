# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from visualize_profile import bar_data, merged_duration, summarize


class ProfileSummaryTest(unittest.TestCase):
    def test_run_selection_keeps_recorded_timestamps(self):
        events = [{"name": "kernel", "ts": 900, "dur": 300}, {"name": "kernel", "ts": 2200, "dur": 100}]
        data = bar_data(events, "GPU", {"start_us": 1000, "duration_ms": 1})
        self.assertEqual(data["base"], [0.9])
        self.assertEqual(data["x"], [0.3])

    def test_overlap_and_clipping_preserve_cpu_origin(self):
        events = [{"name": "model_run", "ts": 1000, "dur": 1000}]
        for start, duration in [(900, 300), (1100, 500), (1800, 400)]:
            events.append(
                {"name": "kernel", "ts": start, "dur": duration, "args": {"clock_alignment": "d3d12_clock_calibration"}}
            )
        result = summarize(events, {"runs": [{"stage": "prefill", "warmup": False}]})
        self.assertEqual(result["runs"][0]["start_us"], 1000)
        self.assertAlmostEqual(result["runs"][0]["gpu_busy_ms"], 0.8)
        self.assertAlmostEqual(result["runs"][0]["gpu_gap_ms"], 0.2)
        self.assertEqual(merged_duration([]), 0)

    def test_approximation_is_not_relabelled(self):
        result = summarize([{"name": "kernel", "ts": 1, "dur": 2, "args": {"clock_alignment": "first_submit"}}], {})
        self.assertEqual(result["alignment"], ["first_submit"])
        with self.assertRaises(ValueError):
            summarize([], {})
        with self.assertRaises(ValueError):
            summarize(
                [{"name": "kernel", "ts": 1, "dur": 2, "args": {"clock_alignment": "first_submit"}}], {"runs": [{}]}
            )


if __name__ == "__main__":
    unittest.main()
