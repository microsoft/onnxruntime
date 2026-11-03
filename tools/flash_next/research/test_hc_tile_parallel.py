import math
from pathlib import Path
import struct
import unittest

from hc_tile_parallel_transform import CLASS, FLAG, SHARED_BYTES, make_program, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class TileParallelTests(unittest.TestCase):
    def test_actual_renderer_preserves_tile_dot_and_both_ordered_folds(self):
        source = SOURCE.read_text()
        start = source.index("class HCPostNormProjectionProgram final")
        stop = source.index("class HCPostNormProjectionF16", start)
        original_program = source[start:stop]
        for shared, scale in ((False, False), (False, True), (True, False), (True, True)):
            original = render(original_program, 0, shared, scale, helper="")
            shader = render(make_program(source), 1, shared, scale, helper="",
                            flag=FLAG, class_name=CLASS)
            begin, end = "  let block = tile * 32u + lane;", "  total += sum;"
            original_dot = original.split(begin)[1].split(end)[0]
            selected_dot = shader.split(begin)[1].split(
                "tile_sums[tile * 256u + logical_lane]")[0]
            self.assertEqual(selected_dot, original_dot.replace("  var sum = f16(0);\n", "") + "}\n")
            self.assertIn("for (var tile = 0u; tile < 10u; tile++) {\n"
                          "  total += f16(tile_sums[tile * 256u + local_idx]);", shader)
            self.assertIn("for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }", shader)
            self.assertEqual(shader.count("local_idx < 256u && h < uniforms.hidden"), 3)
            self.assertIn("if (local_idx < 256u) {\nsums[0u + local_idx]", shader)
            self.assertEqual(shader.count("workgroupBarrier();"), original.count("workgroupBarrier();") + 1)

    def test_unique_full_word_tile_writers_and_original_lane_mapping(self):
        owners = {}
        for thread in range(1024):
            logical_lane = thread % 256
            for tile in range(thread // 256, 10, 4):
                address = tile * 256 + logical_lane
                self.assertNotIn(address, owners)
                owners[address] = thread
                self.assertEqual(logical_lane % 32, thread % 32)
        self.assertEqual(set(owners), set(range(2560)))
        self.assertEqual(SHARED_BYTES, 1024 * 4 + 11520 * 2 + 256 * 2 + 2560 * 4)
        self.assertLessEqual(SHARED_BYTES, 49152)

    def test_finite_half_float_half_roundtrip_is_bitwise(self):
        for bits in range(65536):
            packed = bits.to_bytes(2, "little")
            value, = struct.unpack("<e", packed)
            if math.isfinite(value):
                fp32, = struct.unpack("<f", struct.pack("<f", value))
                self.assertEqual(struct.pack("<e", fp32), packed)

    def test_original_program_baseline_fallback_and_hardware_limits_preserved(self):
        source = SOURCE.read_text()
        result = transform(source)
        start = source.index("class HCPostNormProjectionProgram final")
        stop = source.index("class HCPostNormProjectionF16", start)
        self.assertIn(source[start:stop], result)
        baseline = "    HCPostNormProjectionBaselineNormProgram"
        self.assertEqual(source[source.index(baseline):], result[result.index(baseline):])
        self.assertIn("maxComputeInvocationsPerWorkgroup >= 1024", result)
        self.assertIn("maxComputeWorkgroupStorageSize >= 37888", result)
        self.assertIn("selector == 0 || selector == 1", result)
        self.assertNotIn("}, 0, tile_parallel4)", result)


if __name__ == "__main__":
    unittest.main()
