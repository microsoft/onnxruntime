from pathlib import Path
import unittest

from build_hc_wide_rms_broadcast import CLASS, FLAG, STORAGE_BYTES, make_program, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class WideRmsBroadcastTests(unittest.TestCase):
    def test_original_geometry_and_replay_api_preserved(self):
        source = SOURCE.read_text()
        candidate = make_program(source)
        self.assertNotIn("virtual_idx", candidate)
        self.assertNotIn("h += 128u", candidate)
        self.assertIn("workgroup_idx * 8u + local_idx / 32u", candidate)
        self.assertIn("partials: array<f16, 256>", candidate)
        self.assertIn("width = 128u; width >= 32u", candidate)
        for stream in range(4):
            self.assertIn(f"let rms_total{stream} = subgroupShuffle(reduced{stream}, 0u)", candidate)
        modified = transform(source)
        self.assertEqual(modified.count(".SetWorkgroupSize(256).SetDispatchGroupSize(41);"), 1)
        self.assertNotIn(".SetWorkgroupSize(rms_broadcast", modified)
        self.assertNotIn("}, 0, rms_broadcast);", modified)
        self.assertIn("static const int selector", modified)
        self.assertIn("subgroupMinSize == 32", modified)
        self.assertIn("subgroupMaxSize == 32", modified)
        self.assertEqual(STORAGE_BYTES, 1024 * 4 + 11520 * 2 + 256 * 2)

    def test_projection_and_output_code_unchanged(self):
        source = SOURCE.read_text()
        start = source.index("class HCPostNormProjectionProgram final")
        stop = source.index("class HCPostNormProjectionF16", start)
        original = source[start:stop]
        candidate = make_program(source)
        marker = '         << "let lane = local_idx % 32u;\\n"'
        self.assertEqual(original[original.index(marker):], candidate[candidate.index(marker):])

    def test_rendered_original_staging_and_projection_preserved(self):
        source = SOURCE.read_text()
        start = source.index("class HCPostNormProjectionProgram final")
        stop = source.index("class HCPostNormProjectionF16", start)
        program = make_program(source)
        for shared in (False, True):
            for shared_scale in (False, True):
                original = render(source[start:stop], 0, shared, shared_scale, "", FLAG)
                candidate = render(program, 1, shared, shared_scale, "", FLAG, class_name=CLASS)
                before = "var sum0 = 0.0;"
                after = "for (var h = local_idx; h < uniforms.hidden; h += 256u)"
                self.assertEqual(original[:original.index(before)], candidate[:candidate.index(before)])
                original_norm = original.index(after, original.index("let inv_rms0"))
                candidate_norm = candidate.index(after, candidate.index("let inv_rms0"))
                self.assertEqual(original[original_norm:], candidate[candidate_norm:])


if __name__ == "__main__":
    unittest.main()
