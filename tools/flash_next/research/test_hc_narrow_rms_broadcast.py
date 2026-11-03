import unittest
from pathlib import Path

import numpy as np

from build_hc_narrow_rms_broadcast import CLASS, FLAG, make_program, transform
from build_hc_narrow_rms_broadcast_v2 import transform as v2_transform
from test_hc_projection_inner_unroll import render


class NarrowRmsBroadcastTests(unittest.TestCase):
    def test_rendered_shader_and_capability_fallback(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/"
                      "build-v2/hc_post_norm_projection.cc").read_text()
        for shared in (False, True):
            for shared_scale in (False, True):
                code = render(make_program(source), 0, shared, shared_scale, "", FLAG,
                              class_name=CLASS, constructor_args=(shared, shared_scale))
                self.assertIn("virtual_idx < 256u; virtual_idx += 128u", code)
                self.assertIn("width >= 32u", code)
                self.assertEqual(code.count("subgroupShuffle("), 24)
                self.assertNotIn("subgroup_index == 0u", code)
                self.assertIn("inverseSqrt(rms_total0 /", code)
                self.assertIn("workgroup_idx * 4u + local_idx / 32u", code)
                self.assertIn("partials: array<f16, 128>", code)
                self.assertEqual(code.count("workgroupBarrier();"), 5)
                self.assertIn("sum += dot(x0, w0) + dot(x1, w1)", code)
        candidate = transform(source)
        self.assertIn("subgroupMinSize == 32", candidate)
        self.assertIn("subgroupMaxSize == 32", candidate)
        self.assertIn("context.HasFeature(wgpu::FeatureName::Subgroups)", candidate)
        self.assertIn(f'static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}"', candidate)
        self.assertIn("SetWorkgroupSize(rms_broadcast == 1 ? 128 : 256)", candidate)
        self.assertIn("SetDispatchGroupSize(rms_broadcast == 1 ? 81 : 41)", candidate)
        self.assertIn("}, 0, rms_broadcast);", candidate)
        self.assertIn("HCPostNormProjectionProgram program{shared_, shared_scale}", candidate)
        v2 = v2_transform(source)
        self.assertNotIn("}, 0, rms_broadcast);", v2)
        self.assertIn("static const int selector", v2)
        self.assertEqual(v2, candidate.replace("}, 0, rms_broadcast);", "});", 1))

    def test_cpu_tree_and_all_subgroup_broadcast_bits(self):
        rng = np.random.default_rng(90210)
        for iteration in range(200):
            partials = np.ldexp(rng.random((4, 256), dtype=np.float32),
                               rng.integers(-30, 30, (4, 256))).astype(np.float32)
            if iteration == 0:
                partials.fill(0)
            reference = partials.copy()
            for width in (128, 64, 32, 16, 8, 4, 2, 1):
                reference[:, :width] += reference[:, width:2 * width]
            shared = partials.copy()
            for width in (128, 64, 32):
                shared[:, :width] += shared[:, width:2 * width]
            for subgroup in range(4):
                reduced = shared[:, :32].copy()
                for width in (16, 8, 4, 2, 1):
                    other = np.roll(reduced, -width, axis=1)
                    reduced[:, :width] += other[:, :width]
                broadcast = np.repeat(reduced[:, :1], 32, axis=1)
                expected = np.repeat(reference[:, :1], 32, axis=1)
                self.assertTrue(np.array_equal(broadcast.view(np.uint32), expected.view(np.uint32)),
                                (iteration, subgroup))


if __name__ == "__main__":
    unittest.main()
