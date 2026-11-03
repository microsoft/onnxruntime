from pathlib import Path
import unittest

import numpy as np

from build_hc_projection_shuffle import FLAG, shuffled_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/"
              "hc_post_norm_projection.cc")


class HCProjectionShuffleTests(unittest.TestCase):
    def test_normalization_dot_products_and_original_paths_preserved(self):
        source = SOURCE.read_text()
        original = source[source.index("class HCPostNormProjectionProgram final"):
                          source.index("class HCPostNormProjectionF16")]
        candidate = shuffled_program(source)
        normalization_start = "    for (int stream = 0; stream < 4; ++stream) {"
        normalization_stop = '    code << "}\\nworkgroupBarrier();\\n"\n'
        self.assertEqual(original.split(normalization_start, 1)[1].split(normalization_stop, 1)[0],
                         candidate.split(normalization_start, 1)[1].split(normalization_stop, 1)[0])
        dot_start = '         << "var total = f16(0);\\n"'
        dot_stop = '         << "}\\n}\\n"'
        self.assertEqual(original.split(dot_start, 1)[1].split(dot_stop, 1)[0],
                         candidate.split(dot_start, 1)[1].split(dot_stop, 1)[0])
        actual = transform(source)
        self.assertIn(original, actual)
        fallback = source[source.index("    HCPostNormProjectionBaselineNormProgram norm"):]
        self.assertIn(fallback, actual)
        self.assertEqual(actual.count(".SetWorkgroupSize(256).SetDispatchGroupSize(41)"), 1)

    def test_subgroup_identity_guards_and_replay_keys(self):
        actual = transform(SOURCE.read_text())
        candidate = shuffled_program(SOURCE.read_text())
        for expected in (
                "context.HasFeature(wgpu::FeatureName::Subgroups)",
                'context.AdapterInfo().vendor == std::string_view{"nvidia"}',
                "subgroupMinSize == 32", "subgroupMaxSize == 32",
                f'"{FLAG}", 0', f'"{FLAG} must be 0 or 1"',
                "}, 0, projection_shuffle);"):
            self.assertIn(expected, actual)
        self.assertIn("let lane = sg_id;", candidate)
        self.assertIn("workgroup_idx * 8u + subgroup_index", candidate)
        self.assertNotIn("partials", candidate)
        self.assertIn("HCPostNormProjectionShuffleF16", candidate)
        shuffle = candidate.index("subgroupShuffle(shuffle_total, i)")
        writer = candidate.index("if (lane == 0u && col < 324u)")
        self.assertLess(shuffle, writer)
        self.assertIn("for (var i = 0u; i < 32u; i++)", candidate)

    def test_float32_transport_preserves_every_finite_half_bit_pattern(self):
        halves = np.arange(65536, dtype=np.uint16).view(np.float16)
        halves = halves[np.isfinite(halves)]
        restored = halves.astype(np.float32).astype(np.float16)
        np.testing.assert_array_equal(halves.view(np.uint16), restored.view(np.uint16))

    def test_serial_half_addition_order_is_unchanged(self):
        random = np.random.default_rng(42)
        for _ in range(128):
            partials = random.uniform(-8, 8, size=(8, 32)).astype(np.float16)
            original = np.zeros(8, dtype=np.float16)
            candidate = np.zeros(8, dtype=np.float16)
            transport = partials.astype(np.float32)
            for lane in range(32):
                original += partials[:, lane]
                candidate += transport[:, lane].astype(np.float16)
            np.testing.assert_array_equal(original.view(np.uint16), candidate.view(np.uint16))


if __name__ == "__main__":
    unittest.main()
