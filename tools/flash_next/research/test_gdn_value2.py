from pathlib import Path
import unittest

from build_gdn_value2 import transform, transform_compare
from build_gdn_value2_capture import transform as transform_capture


BASE = Path("/home/sayanshaw/repos")
COMPONENT = BASE / "qwen38-int4-gdn-decode-view/build-original"
HEADER = BASE / "onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/gated_delta_net.h"


class GdnValue2Tests(unittest.TestCase):
    def test_original_shader_method_and_fallback_setup_are_preserved(self):
        source = (COMPONENT / "gated_delta_net.cc").read_text()
        actual = transform(source, HEADER.read_text())
        start = source.index("Status GatedDeltaNetProgram::GenerateShaderCode")
        stop = source.index("Status GatedDeltaNetNormalizeProgram::GenerateShaderCode", start)
        self.assertIn(source[start:stop], actual)
        self.assertIn("WGSL_TEMPLATE_PARAMETER(value_channels_per_workgroup, 2)", actual)
        self.assertIn("return run_program(program, value_tiles);", actual)
        self.assertIn("* dispatch_value_tiles)", actual)
        self.assertIn("use_packed_qkv && dk == 128 && dv == 128", actual)
        self.assertIn("total_tokens == 1 && batch == 1", actual)
        self.assertIn("query->IsDataType<MLFloat16>()", actual)
        self.assertIn("context.DeviceLimits().maxComputeWorkgroupsPerDimension", actual)
        def addresses(channels):
            tiles = 128 // channels
            return [(head * 128 + (group % tiles) * channels + offset) * 128 + key
                    for group in range(48 * tiles)
                    for head in (group // tiles,)
                    for offset in range(channels)
                    for key in (0, 63, 127)]

        original, candidate = addresses(4), addresses(2)
        self.assertEqual(set(original), set(candidate))
        self.assertEqual(len(candidate), len(set(candidate)))
        self.assertEqual(len(candidate), 48 * 128 * 3)

    def test_comparator_preserves_all_output_checks_and_rejects_invalid_flag(self):
        actual = transform_compare((COMPONENT / "native_compare.cpp").read_text())
        self.assertIn("output < names.size()", actual)
        self.assertIn("std::memcmp(", actual)
        self.assertIn("std::isfinite(av)", actual)
        self.assertIn("run < 9", actual)
        self.assertIn('set_mid("2")', actual)
        self.assertIn('{"maxNumPendingDispatches", pending}', actual)
        self.assertIn("ORT_TEST_TOGGLE_GDN_VALUE2", actual)

    def test_capture_uses_fixed_cpu_outputs_for_both_sessions(self):
        source = transform_compare((COMPONENT / "native_compare.cpp").read_text())
        actual = transform_capture(source)
        self.assertIn("baseline_binding.BindOutput(names[output], expected_buffers.back())", actual)
        self.assertIn("fused_binding.BindOutput(names[output], actual_buffers.back())", actual)
        self.assertIn("binding.SynchronizeOutputs()", actual)
        self.assertIn("auto expected = bound_run(baseline, baseline_binding)", actual)
        self.assertIn("auto actual = bound_run(fused, fused_binding)", actual)
        self.assertIn("std::memcmp(", actual)
        self.assertIn("output < names.size()", actual)


if __name__ == "__main__":
    unittest.main()
