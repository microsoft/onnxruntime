from collections import Counter
from pathlib import Path
import struct
import unittest

from dense_tile_parallel3_transform import CLASS, FLAG, NAME, ORIGINAL_PROGRAM, PROGRAM, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8-double-buffer/build-v1/matmul_nbits.cc")


class DenseTileParallelThreeTests(unittest.TestCase):
    def test_actual_renderer_retains_both_dot_sequences_and_ordered_folds(self):
        shader = render(PROGRAM, 1, False, False, helper="", flag=FLAG,
                        class_name=CLASS, constructor_args=())
        original = render(ORIGINAL_PROGRAM, 0, False, False, helper="", flag=FLAG,
                          class_name="DenseColumns8DoubleBufferProgram", constructor_args=())
        original_dot = original.split("    for (var word = 0u;", 1)[1].split(
            "    partials[", 1)[0].rstrip()
        selected_dot = shader.split("    for (var word = 0u;", 1)[1].split(
            "  }\n  tile_sums", 1)[0].rstrip()
        self.assertEqual(selected_dot.replace("buffer * 256u", "(tile % 2u) * 256u"), original_dot)
        self.assertIn("for (var tile = 0u; tile * 32u < uniforms.blocks; tile++)", shader)
        self.assertIn("if (tile * 32u + stream < uniforms.blocks)", shader)
        self.assertIn("partial0 += pair.x;\n      partial1 += pair.y;", shader)
        self.assertIn("for (var stream = 0u; stream < 32u; stream++)", shader)
        self.assertIn("value += pair[local_idx / 4u]", shader)
        self.assertNotIn("partials[", shader)
        self.assertEqual(shader.count("workgroupBarrier();"), 3)

    def test_tile_words_activation_vectors_and_reused_partial_words_have_unique_owners(self):
        for blocks in (80, 192):
            stores, vectors = Counter(), Counter()
            wave = 0
            while wave * 96 < blocks:
                for physical in range(384):
                    logical, worker = physical % 128, physical // 128
                    tile = wave * 3 + worker
                    stores[tile * 128 + logical] += 1
                    for index in range(logical, 256, 128):
                        vectors[((wave % 2) * 3 + worker) * 256 + index] += 1
                wave += 1
            tiles = (blocks + 31) // 32
            self.assertEqual(stores, Counter({index: 1 for index in range(tiles * 128)}))
            self.assertEqual(vectors, Counter({index: 1 for index in range(wave * 3 * 256)}))
            for logical in range(128):
                reads = [tile * 128 + logical for tile in range(tiles)]
                self.assertEqual([slot % 128 for slot in reads], [logical] * tiles)
        readers = [(column % 4) * 32 + stream for column in range(8) for stream in range(32)]
        self.assertEqual(Counter(readers), Counter({index: 2 for index in range(128)}))

    def test_original_default_fallbacks_and_scoped_replay_key_are_preserved(self):
        source = SOURCE.read_text()
        candidate = transform(source)
        self.assertIn(ORIGINAL_PROGRAM, candidate)
        for text in (FLAG + " must be 0 or 1", "(K_ == 2560 || K_ == 6144)",
                     "maxComputeWorkgroupSizeX >= 384", "maxComputeInvocationsPerWorkgroup >= 384",
                     "maxComputeWorkgroupStorageSize >= 15360",
                     ".SetWorkgroupSize(384).SetDispatchGroupSize(N_ / 8)",
                     "columns8 ? (use_tile_parallel3 ? 8 : 7)", NAME):
            self.assertIn(text, candidate)
        original_tail = source[source.index("\n/**\n * @brief Applies"): ]
        self.assertTrue(candidate.endswith(original_tail))
        original_branch = source[source.index("    if (columns8) {"):source.index(
            "    if (dense_bitcast_route && block32_mode == 1)")]
        self.assertIn(original_branch, candidate)

    def test_packed_pair_reuse_matches_original_both_levels_of_half_additions(self):
        def bits(value):
            return struct.unpack("<H", struct.pack("<e", value))[0]

        def value(raw):
            return struct.unpack("<e", struct.pack("<H", raw))[0]

        def add(a, b):
            return bits(value(a) + value(b))

        choices = [bits(number) for number in (-0.0, 0.0, 0.03125, -0.0625, 2**-24, -(2**-24))]
        for blocks in (80, 192):
            tiles = (blocks + 31) // 32
            scratch, original = {}, {}
            for tile in range(tiles):
                for logical in range(128):
                    pair = []
                    for component in range(2):
                        column, stream = logical // 32 + component * 4, logical % 32
                        raw = choices[(tile * 13 + logical * 3 + component) % len(choices)]
                        if tile * 32 + stream < blocks:
                            key = (column, stream)
                            original[key] = add(original.get(key, bits(0)), raw)
                        else:
                            raw = 0x7E00
                        pair.append(raw)
                    scratch[tile * 128 + logical] = pair[0] | pair[1] << 16
            for logical in range(128):
                partials = [bits(0), bits(0)]
                for tile in range(tiles):
                    if tile * 32 + logical % 32 < blocks:
                        word = scratch[tile * 128 + logical]
                        partials = [add(partials[0], word & 0xFFFF),
                                    add(partials[1], word >> 16)]
                scratch[logical] = partials[0] | partials[1] << 16
            for column in range(8):
                expected, observed = bits(0), bits(0)
                for stream in range(32):
                    expected = add(expected, original[(column, stream)])
                    word = scratch[(column % 4) * 32 + stream]
                    observed = add(observed, (word >> (16 * (column // 4))) & 0xFFFF)
                self.assertEqual(observed, expected)

    def test_native_scope_matches_actual_gpu_filter_and_original_byte_reference(self):
        from build_dense_tile_parallel3 import ROOT, native_source
        source = native_source((ROOT / "dense_paired_matrix_compare.cpp").read_text())
        self.assertIn("constexpr size_t rounds = 36;", source)
        self.assertIn("round * 4 + repeat == 16", source)
        self.assertIn('{"maxNumPendingDispatches", "64"}', source)
        self.assertIn("measured_end.ru_majflt != measured_start.ru_majflt", source)
        self.assertIn("SCOPE warmups=16 measured=128 major_faults=0", source)
        self.assertIn("std::memcmp(&expected, &observed[i], sizeof(expected))", source)


if __name__ == "__main__":
    unittest.main()
