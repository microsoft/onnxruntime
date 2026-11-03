from pathlib import Path
import unittest

from build_hc_bitcast_unpack import transform


class HCBitcastTests(unittest.TestCase):
    def test_scoped_shader_change_and_variant_cache(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/"
                      "build-v2/hc_post_norm_projection.cc").read_text()
        result = transform(source)
        program_start = source.index("class HCPostNormProjectionProgram final")
        self.assertEqual(source[:program_start], result[:program_start])
        self.assertIn("HCPostNormProjectionBitcastF16", result)
        self.assertIn("}, 0, bitcast);", result)
        self.assertIn("ORT_WEBGPU_HC_POST_NORM_BITCAST must be 0 or 1", result)
        for line in ('for (var tile = 0u; tile < 10u; tile++)',
                     'total += sum;', 'value += partials[local_idx + i];',
                     '.SetWorkgroupSize(256).SetDispatchGroupSize(41)'):
            self.assertEqual(source.count(line), result.count(line))
        old_fields = source[source.index(" protected:", source.index("class HCPostNormProjectionF16")):]
        new_fields = result[result.index(" protected:", result.index("class HCPostNormProjectionF16")):]
        self.assertEqual(old_fields, new_fields)


if __name__ == "__main__":
    unittest.main()
