from pathlib import Path
import unittest

from build_qsa_score16 import DISPATCH_ANCHOR, FLAG, STORAGE_BYTES, transform
from run_qsa_parallel_map_native import packed_score_eligible


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qsa-indexer-stack/build-v1/sparse_attention_indexer.cc")


class Score16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = SOURCE.read_text()
        cls.result = transform(cls.original)
        start = cls.result.index("Status QsaPackedScore16Program::GenerateShaderCode")
        stop = cls.result.index("\nclass QsaPackedScoreProgram", start)
        cls.shader = cls.result[start:stop]

    def test_scoped_default_off_and_original_fallback(self):
        self.assertIn(f'("{FLAG}", 0)', self.result)
        self.assertIn("score16 == 0 || score16 == 1", self.result)
        self.assertIn("score16 == 1 && (packed_score == 1 || indexer_stack == 1)", self.result)
        self.assertIn("num_heads <= 4 && score_stride >= 8192", self.result)
        self.assertIn("num_heads <= 4 && score_stride >= 2048",
                      transform(self.original, minimum_stride=2048))
        self.assertIn(f"maxComputeWorkgroupStorageSize >= {STORAGE_BYTES}", self.result)
        self.assertEqual(STORAGE_BYTES, 8516)
        self.assertLess(self.result.index(f"{FLAG} must be 0 or 1"),
                        self.result.index('const int indexer_stack ='))
        for start_marker, end_marker in (
                ("class QsaPackedScoreProgram final", "\nStatus SparseAttentionIndexerQsaSelectProgram::GenerateShaderCode"),
                (DISPATCH_ANCHOR, "    if (use_parallel_rank && phase == QsaCachePhase::Select) {")):
            start = self.original.index(start_marker)
            stop = self.original.index(end_marker, start)
            self.assertIn(self.original[start:stop], self.result)

    def test_unique_programs_and_shared_slots(self):
        for name in ("QsaPackedScoreProgram", "QsaPackedScore16Program",
                     "SparseAttentionIndexerQsaSelectProgram"):
            self.assertEqual(self.result.count(f"Status {name}::GenerateShaderCode"), 1)
        self.assertIn('Program{"QsaPackedScore16"}', self.result)
        self.assertIn("if (local_idx == 0u) { visible_count_shared", self.shader)
        self.assertNotIn("if (local_lane == 0u) { visible_count_shared", self.shader)
        self.assertIn("head_scores[group * 4u + local_lane]", self.shader)
        self.assertIn("key_inverse_rms: array<f32, 16>", self.shader)
        slots = [g * 4 + h for g in range(16) for h in range(4)]
        self.assertEqual(slots, list(range(64)))

    def test_serial_arithmetic_and_barriers_preserved(self):
        for expression in (
            "square_sum += pooled_key[group * 128u + d] * pooled_key[group * 128u + d]",
            "square_sum += value * value",
            "dot += query_value(row, local_lane, d, inv_rms) * pooled_key[group * 128u + d]",
            "score += head_scores[group * 4u + head]",
        ):
            self.assertIn(expression, self.shader)
        start = self.original.index("    if (phase_ == QsaCachePhase::Score) {",
                                    self.original.index("Status QsaPackedScoreProgram"))
        stop = self.original.index("    if (phase_ == QsaCachePhase::Map) {", start)
        new_start = self.shader.index("    if (phase_ == QsaCachePhase::Score) {")
        new_stop = self.shader.index("    if (phase_ == QsaCachePhase::Map) {", new_start)
        self.assertEqual(self.original[start:stop].count("workgroupBarrier()"),
                         self.shader[new_start:new_stop].count("workgroupBarrier()"))

    def test_tail_and_element_coverage(self):
        for stride in (1, 15, 16, 17, 2048, 2051, 8192, 8193, 16384):
            actual = [wg * 16 + g for wg in range((stride + 15) // 16)
                      for g in range(16) if wg * 16 + g < stride]
            self.assertEqual(actual, list(range(stride)))
        for width in (4, 6, 32, 128):
            actual = [d for lane in range(4) for d in range(lane, width, 4)]
            self.assertEqual(sorted(actual), list(range(width)))

    def test_native_eligibility_uses_actual_head_and_batch_dimensions(self):
        for batch, rows, heads, stride, expected in (
                (1, 1, 4, 2048, False), (2, 1, 4, 8192, False),
                (1, 16, 4, 8192, False), (1, 1, 5, 8192, False),
                (1, 1, 4, 8192, True), (1, 1, 4, 8191, False), (1, 1, 1, 8193, True)):
            event = dict(cat="Kernel", name="node&QsaPackedScore4",
                         args=dict(shapes=f"inputs[0] = {{{batch},{rows},640}} outputs[2] = {{{batch * rows},{stride}}}",
                                   cache_key=f"QsaPackedScore4[10|{heads}|128]"))
            self.assertEqual(packed_score_eligible([event] * 18, minimum_stride=8192,
                maximum_heads=4, program_name="QsaPackedScore4"), expected)

    def test_padding_changes_only_shared_key_layout(self):
        plain = transform(self.original, minimum_stride=2048)
        padded = transform(self.original, minimum_stride=2048, pad_keys=True)
        self.assertIn('Program{"QsaPackedScore16Padded"}', padded)
        self.assertIn("maxComputeWorkgroupStorageSize >= 8772", padded)
        normalized = padded.replace('Program{"QsaPackedScore16Padded"}', 'Program{"QsaPackedScore16"}').replace(
            "16 * (kQsaSharedHeadSize + 4)", "16 * kQsaSharedHeadSize").replace(
            "group * 132u", "group * 128u").replace(
            "maxComputeWorkgroupStorageSize >= 8772", "maxComputeWorkgroupStorageSize >= 8516")
        self.assertEqual(normalized, plain)
        slots = [g * 132 + d for g in range(16) for d in range(128)]
        self.assertEqual(len(set(slots)), 16 * 128)
        self.assertLess(max(slots), 16 * 132)

    def test_query_cache_preserves_serial_rms_and_query_value_helper(self):
        cached = transform(self.original, minimum_stride=2048, pad_keys=True, cache_query=True)
        self.assertIn('Program{"QsaPackedScore16QueryCache"}', cached)
        self.assertIn("maxComputeWorkgroupStorageSize >= 10852", cached)
        start = cached.index("Status QsaPackedScore16Program::GenerateShaderCode")
        stop = cached.index("\nclass QsaPackedScoreProgram", start)
        shader = cached[start:stop]
        self.assertIn("if (local_idx < uniforms.num_heads)", shader)
        self.assertIn("square_sum += value * value", shader)
        self.assertIn("query_inverse_rms[local_idx] = inverseSqrt", shader)
        self.assertIn("query_values[head * 129u + d] = query_value(row, head, d, query_inverse_rms[head])", shader)
        self.assertIn("dot += query_values[local_lane * 129u + d] * pooled_key[group * 132u + d]", shader)
        self.assertNotIn("let inv_rms =", shader)
        first = shader.index("    if (phase_ == QsaCachePhase::Score) {")
        last = shader.index("    if (phase_ == QsaCachePhase::Map) {", first)
        self.assertEqual(shader[first:last].count("workgroupBarrier()"), 8)
        for heads in (1, 2, 4):
            for width in (4, 6, 32, 128):
                elements = [i for lane in range(64) for i in range(lane, heads * width, 64)]
                self.assertEqual(sorted(elements), list(range(heads * width)))
                slots = [(i // width) * 129 + i % width for i in elements]
                self.assertEqual(len(set(slots)), heads * width)
                self.assertLess(max(slots), 4 * 129)

    def test_wide_group_mapping_and_limits(self):
        wide = transform(self.original, minimum_stride=2048, pad_keys=True,
                         cache_query=True, wide_group=True)
        self.assertIn('Program{"QsaPackedScore16QueryWide"}', wide)
        self.assertIn("maxComputeWorkgroupStorageSize >= 11620", wide)
        self.assertIn("maxComputeWorkgroupSizeX >= 256", wide)
        self.assertIn("maxComputeInvocationsPerWorkgroup >= 256", wide)
        self.assertIn("SetWorkgroupSize(256)", wide)
        self.assertIn("local_idx / 16u", wide)
        self.assertIn("local_idx % 16u", wide)
        self.assertIn("i += 256u", wide)
        self.assertIn("group * 144u + d", wide)
        slots = [g * 144 + d for g in range(16) for d in range(128)]
        self.assertEqual(len(set(slots)), 16 * 128)
        self.assertLess(max(slots), 16 * 144)


if __name__ == "__main__":
    unittest.main()
