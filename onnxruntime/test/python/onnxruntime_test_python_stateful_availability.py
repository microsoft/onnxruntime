# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Exercise the stateful feature gate against the build's Python extension."""

import os
import unittest
from pathlib import Path

import onnxruntime as ort
from onnxruntime import stateful
from onnxruntime.capi import onnxruntime_pybind11_state as native


class StatefulAvailabilityTest(unittest.TestCase):
    def test_build_configuration(self):
        output_dir = Path(os.environ["ORT_BUILD_OUTPUT_DIR"]).resolve()
        self.assertTrue(Path(ort.__file__).resolve().is_relative_to(output_dir))
        expected_setting = os.environ["ORT_EXPECT_STATEFUL"]
        self.assertIn(expected_setting, ("0", "1"))
        expected = expected_setting == "1"
        self.assertEqual(native._stateful.available, expected)
        self.assertEqual(stateful.SamplerConfig().algorithm, stateful.SamplingAlgorithm.GREEDY)
        if expected:
            sampler = stateful.Sampler()
            sampler.close()
        else:
            with self.assertRaisesRegex(native.NotImplemented, "Stateful execution is unavailable"):
                stateful.Sampler()


if __name__ == "__main__":
    unittest.main()
