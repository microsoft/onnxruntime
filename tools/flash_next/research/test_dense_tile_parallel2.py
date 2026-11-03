from collections import Counter
from pathlib import Path
import unittest

from dense_tile_parallel2_transform import CLASS, FLAG, NAME, PROGRAM, SHARED_BYTES, transform
from dense_tile_parallel3_transform import ORIGINAL_PROGRAM, PROGRAM as THREE_PROGRAM


class DenseTileParallelTwoTests(unittest.TestCase):
    def test_only_geometry_changes_and_original_half_dot_and_fold_order_remain(self):
        restored = PROGRAM.replace(NAME, "DenseTileParallel3Packed")
        for old, new in (("tile_a: array<vec4<f16>, 1024>", "tile_a: array<vec4<f16>, 1536>"),
                         ("wave * 64u < uniforms.blocks", "wave * 96u < uniforms.blocks"),
                         ("wave * 2u + tile_worker", "wave * 3u + tile_worker"),
                         ("(wave % 2u) * 2u + tile_worker", "(wave % 2u) * 3u + tile_worker")):
            self.assertEqual(restored.count(old), 1)
            restored = restored.replace(old, new)
        self.assertEqual(restored, THREE_PROGRAM)
        self.assertEqual(SHARED_BYTES, 1024 * 8 + 768 * 4)

    def test_full_word_owners_and_reused_activation_buffers_are_bounded_without_races(self):
        for blocks in (80, 192):
            waves = (blocks + 63) // 64
            tiles = (blocks + 31) // 32
            words = Counter()
            previous_vectors = set()
            for wave in range(waves):
                vectors = Counter()
                for physical in range(256):
                    logical, worker = physical % 128, physical // 128
                    tile = wave * 2 + worker
                    words[tile * 128 + logical] += 1
                    for index in range(logical, 256, 128):
                        vectors[((wave % 2) * 2 + worker) * 256 + index] += 1
                self.assertTrue(all(count == 1 for count in vectors.values()))
                self.assertEqual(len(vectors), 512)
                self.assertLess(max(vectors), 1024)
                self.assertTrue(previous_vectors.isdisjoint(vectors))
                previous_vectors = set(vectors)
            self.assertEqual(words, Counter({index: 1 for index in range(waves * 2 * 128)}))
            self.assertLess(max(words), 768)
            reads = {tile * 128 + logical for tile in range(tiles) for logical in range(128)}
            self.assertTrue(reads.issubset(words))
            self.assertEqual(len(reads), tiles * 128)
            if blocks == 80:
                self.assertEqual(set(words) - reads, set(range(384, 512)))
            for logical in range(128):
                self.assertEqual([slot % 128 for slot in
                                  (tile * 128 + logical for tile in range(tiles))], [logical] * tiles)

    def test_original_defaults_and_fallbacks_are_preserved_with_fresh_selector(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8-double-buffer/build-v1/matmul_nbits.cc").read_text()
        candidate = transform(source)
        self.assertIn(ORIGINAL_PROGRAM, candidate)
        for text in (FLAG + " must be 0 or 1", "(K_ == 2560 || K_ == 6144)",
                     "maxComputeWorkgroupSizeX >= 256", "maxComputeInvocationsPerWorkgroup >= 256",
                     "maxComputeWorkgroupStorageSize >= 11264", CLASS + " program;",
                     ".SetWorkgroupSize(256).SetDispatchGroupSize(N_ / 8)",
                     "columns8 ? (use_tile_parallel2 ? 8 : 7)"):
            self.assertIn(text, candidate)
        self.assertNotIn("ORT_WEBGPU_DENSE_TILE_PARALLEL3", candidate)
        self.assertTrue(candidate.endswith(source[source.index("\n/**\n * @brief Applies"):]))


if __name__ == "__main__":
    unittest.main()
