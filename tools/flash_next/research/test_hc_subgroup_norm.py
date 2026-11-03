from pathlib import Path
import unittest

import numpy as np

from build_hc_subgroup_norm import FLAG, transform


class HCSubgroupNormTests(unittest.TestCase):
    def test_scoped_reduction_preserves_materialization_and_layout(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/"
                      "hc_post_norm_projection.cc").read_text()
        actual = transform(source)
        start = source.index("class HCPostNormProjectionProgram final")
        self.assertEqual(source[:start], actual[:start])
        first = source.index('         << "for (var width = 128u;', start)
        self.assertEqual(source[start:first].split("    const auto& streams", 1)[1],
                         actual[actual.index("class HCPostNormProjectionProgram final"):
                                actual.index('         << (subgroup_norm_ ?')].split("    const auto& streams", 1)[1])
        self.assertIn("HCPostNormProjectionSubgroupNormF16", actual)
        self.assertIn("stream * 256 + 32", actual)
        self.assertIn("stream * 256 + (subgroup_norm_ ? 32 : 0)", actual)
        self.assertIn("subgroupShuffle(reduced", actual)
        self.assertIn("if (subgroup_index == 0u)", actual)
        self.assertIn("if (sg_id == 0u) { sums[", actual)
        self.assertIn("subgroupMinSize == 32", actual)
        self.assertIn("subgroupMaxSize == 32", actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("}, 0, subgroup_norm);", actual)
        self.assertEqual(source[source.index(" protected:"):], actual[actual.index(" protected:"):])
        for expression in ("sum += dot(x0, w0) + dot(x1, w1)", "total += sum;",
                           "for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }"):
            self.assertEqual(source.count(expression), actual.count(expression))

    def test_identical_float32_addition_tree(self):
        random = np.random.default_rng(42)
        for _ in range(256):
            values = random.uniform(-1000, 1000, size=(4, 256)).astype(np.float32)
            original = values.copy()
            for width in (128, 64, 32, 16, 8, 4, 2, 1):
                original[:, :width] += original[:, width:2 * width]
            candidate = values.copy()
            for width in (128, 64, 32):
                candidate[:, :width] += candidate[:, width:2 * width]
            lanes = candidate[:, :32].copy()
            for width in (16, 8, 4, 2, 1):
                other = np.roll(lanes, -width, axis=1)
                lanes[:, :width] += other[:, :width]
            np.testing.assert_array_equal(original[:, 0].view(np.uint32), lanes[:, 0].view(np.uint32))


if __name__ == "__main__":
    unittest.main()
