from pathlib import Path
import unittest

from build_qsa_packed_score import FLAG, transform
from run_qsa_parallel_map_native import packed_score_eligible


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qsa-parallel-map/build-v1/sparse_attention_indexer.cc")


class PackedScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = SOURCE.read_text()
        cls.result = transform(cls.original)
        start = cls.result.index("Status QsaPackedScoreProgram::GenerateShaderCode")
        stop = cls.result.index("Status SparseAttentionIndexerQsaSelectProgram::GenerateShaderCode", start)
        cls.shader = cls.result[start:stop]

    def test_default_off_and_eligibility(self):
        self.assertIn(f'("{FLAG}", 0)', self.result)
        self.assertIn("packed_score == 0 || packed_score == 1", self.result)
        self.assertIn("rows == 1", self.result)
        self.assertIn("num_heads <= 16 && score_stride >= 2048", self.result)
        self.assertIn('Program{"QsaPackedScore4"}', self.result)

    def test_original_shader_is_frozen(self):
        start = self.original.index("Status SparseAttentionIndexerQsaSelectProgram::GenerateShaderCode")
        stop = self.original.index("\nStatus ", start + 1)
        self.assertIn(self.original[start:stop], self.result)

    def test_shared_slots_are_disjoint_and_visible_count_has_one_writer(self):
        self.assertIn("if (local_idx == 0u) { visible_count_shared", self.shader)
        self.assertNotIn("if (local_lane == 0u) { visible_count_shared", self.shader)
        self.assertIn("key_inverse_rms[group] =", self.shader)
        self.assertIn("pooled_key[group * 128u + d + half]", self.shader)
        self.assertIn("head_scores[group * 64u + head]", self.shader)
        self.assertIn("key_inverse_rms: array<f32, 4>", self.shader)

    def test_serial_arithmetic_and_barriers_are_preserved(self):
        for expression in (
            "square_sum += pooled_key[group * 128u + d] * pooled_key[group * 128u + d]",
            "square_sum += value * value",
            "dot += query_value(row, local_lane, d, inv_rms) * pooled_key[group * 128u + d]",
            "score += head_scores[group * 64u + head]",
        ):
            self.assertIn(expression, self.shader)
        original_start = self.original.index("    if (phase_ == QsaCachePhase::Score) {")
        original_stop = self.original.index("    if (phase_ == QsaCachePhase::Map) {", original_start)
        new_start = self.shader.index("    if (phase_ == QsaCachePhase::Score) {")
        new_stop = self.shader.index("    if (phase_ == QsaCachePhase::Map) {", new_start)
        self.assertEqual(self.original[original_start:original_stop].count("workgroupBarrier()"),
                         self.shader[new_start:new_stop].count("workgroupBarrier()"))

    def test_candidate_coverage_tail_and_local_dimension_coverage(self):
        for stride in (1, 511, 512, 513, 8192, 8193, 16384):
            actual = [wg * 4 + group for wg in range((stride + 3) // 4)
                      for group in range(4) if wg * 4 + group < stride]
            self.assertEqual(actual, list(range(stride)))
        for hidden in (4, 6, 32, 128):
            actual = [d for lane in range(16) for d in range(lane, hidden, 16)]
            self.assertEqual(sorted(actual), list(range(hidden)))

    def test_native_dispatch_eligibility_uses_actual_batch_shape(self):
        for batch, rows, heads, stride, expected in (
                (1, 1, 4, 512, True), (2, 1, 4, 512, False),
                (1, 16, 4, 8192, False), (1, 1, 17, 512, False),
                (1, 1, 16, 511, False), (1, 1, 16, 513, True),
                (1, 1, 4, 2048, True), (1, 1, 4, 8192, True)):
            event = dict(cat="Kernel", name="node&SparseAttentionIndexerQsaScore",
                         args=dict(shapes=f"inputs[0] = {{{batch},{rows},640}} outputs[2] = {{{batch * rows},{stride}}}",
                                   cache_key=f"SparseAttentionIndexerQsaScore[10|{heads}|128]"))
            self.assertEqual(packed_score_eligible([event] * 18), expected)
            self.assertEqual(packed_score_eligible([event] * 18, minimum_stride=2048),
                             expected and stride >= 2048)


if __name__ == "__main__":
    unittest.main()
