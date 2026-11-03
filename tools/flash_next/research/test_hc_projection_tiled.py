from collections import Counter
from pathlib import Path
import unittest

from build_hc_projection_tiled import FLAG, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-projection-silu/build-v1/hc_projection_silu.cc")


class TiledProjectionTests(unittest.TestCase):
    def test_exact_arithmetic_and_reduction_are_retained(self):
        source = transform(SOURCE.read_text())
        self.assertIn("sum += dot(x0, w0) + dot(x1, w1)", source)
        self.assertIn('<< "  total += sum;\\n"', source)
        self.assertIn("for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }", source)
        self.assertIn("tile_a[block * 9u + i * 2u + 1u]", source)
        self.assertIn("for (var tile = 0u; tile < 10u; tile++)", source)

    def test_independent_replay_caches_and_limits(self):
        source = transform(SOURCE.read_text())
        self.assertIn("auto& replay = tiled ? tiled_replay_ : static_replay_", source)
        self.assertIn("maxComputeWorkgroupStorageSize >= 23552", source)
        self.assertIn(f'"{FLAG} must be 0 or 1"', source)
        self.assertIn("if (col < 324u)", source)
        self.assertIn("lane == 0u && col < 324u", source)

    def test_tile_coverage_and_partial_ownership(self):
        self.assertEqual(2880 * 8 + 256 * 2, 23552)
        self.assertEqual({lane + 256 * step for lane in range(256) for step in range(10)},
                         set(range(2560)))
        cells = [(group * 8 + local // 32, local % 32)
                 for group in range(41) for local in range(256)
                 if group * 8 + local // 32 < 324]
        self.assertEqual(len(cells), 324 * 32)
        self.assertEqual(len(set(cells)), len(cells))
        staged = {(index // 8) * 9 + index % 8 for index in range(2560)}
        reads = {block * 9 + offset for block in range(320) for offset in range(8)}
        self.assertEqual(staged, reads)
        self.assertLess(max(staged), 2880)

    def test_padded_vector_stride_distributes_shared_banks(self):
        banks = Counter((lane * 9 * 2 + word) % 32 for lane in range(32) for word in range(2))
        self.assertEqual(len(banks), 32)
        self.assertEqual(max(banks.values()), 2)


if __name__ == "__main__":
    unittest.main()
