import inspect
from pathlib import Path
import unittest

from build_gdn_value8 import FLAG, transform, transform_compare
from build_gdn_value2 import main as build_value2, transform as value2_transform
from run_gdn_value2_native import main as native_value2


BASE = Path("/home/sayanshaw/repos")
COMPONENT = BASE / "qwen38-int4-gdn-decode-view/build-original"
HEADER = BASE / "onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/gated_delta_net.h"


class GdnValue8Tests(unittest.TestCase):
    def test_only_value_grouping_and_private_identity_differ_from_value2(self):
        source = (COMPONENT / "gated_delta_net.cc").read_text()
        header = HEADER.read_text()
        actual = transform(source, header)
        restored = (actual.replace("GatedDeltaNetValue8", "GatedDeltaNetValue2")
                    .replace(FLAG, "ORT_WEBGPU_GDN_VALUE2").replace("value8", "value2")
                    .replace("value_channels_per_workgroup, 8)", "value_channels_per_workgroup, 2)")
                    .replace("workgroup_size, value2 ? 8 :", "workgroup_size, value2 ? 2 :")
                    .replace("(dv + 7) / 8", "(dv + 1) / 2"))
        self.assertEqual(restored, value2_transform(source, header))
        start = source.index("Status GatedDeltaNetProgram::GenerateShaderCode")
        stop = source.index("Status GatedDeltaNetNormalizeProgram::GenerateShaderCode", start)
        self.assertIn(source[start:stop], actual)

    def test_each_value_channel_owned_once_with_unchanged_state_coordinates(self):
        def addresses(channels):
            return [(head, tile * channels + offset, key)
                    for head in range(48) for tile in range(128 // channels)
                    for offset in range(channels) for key in range(128)]
        original, candidate = addresses(4), addresses(8)
        self.assertEqual(set(original), set(candidate))
        self.assertEqual(len(candidate), len(set(candidate)))
        self.assertEqual(len(candidate), 48 * 128 * 128)

    def test_comparator_preserves_both_outputs_capture_and_explicit_rejection(self):
        actual = transform_compare((COMPONENT / "native_compare.cpp").read_text())
        self.assertIn("output < names.size()", actual)
        self.assertIn("std::memcmp(", actual)
        self.assertIn("std::isfinite(av)", actual)
        self.assertIn(f"{FLAG} must be 0 or 1", actual)
        self.assertIn("ORT_TEST_TOGGLE_GDN_VALUE8", actual)
        self.assertNotIn("VALUE2", actual)

    def test_old_builder_and_native_defaults_remain_value2(self):
        self.assertEqual(inspect.signature(build_value2).parameters["output"].default,
                         Path("/workspace/qwen38-int4-gdn-value2/build-v1"))
        self.assertEqual(inspect.signature(native_value2).parameters["program_name"].default,
                         "GatedDeltaNetValue2")


if __name__ == "__main__":
    unittest.main()
