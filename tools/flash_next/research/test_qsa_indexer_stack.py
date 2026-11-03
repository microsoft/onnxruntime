from pathlib import Path
import unittest

from build_qsa_indexer_stack import FLAG, transform


class IndexerStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path("/home/sayanshaw/repos/qwen38-int4-qsa-parallel-map/build-v1/sparse_attention_indexer.cc")
        cls.result = transform(source.read_text())

    def test_stack_is_default_off_and_validated_before_copy(self):
        self.assertIn(f'("{FLAG}", 0)', self.result)
        self.assertIn("indexer_stack == 0 || indexer_stack == 1", self.result)
        self.assertLess(self.result.index(f'("{FLAG}", 0)'),
                        self.result.index('("ORT_WEBGPU_QSA_CONCAT_VECTOR", 0)'))

    def test_both_children_are_enabled_without_bypassing_validation(self):
        self.assertIn("(concat_vector == 1 || indexer_stack == 1)", self.result)
        self.assertIn("(packed_score == 1 || indexer_stack == 1)", self.result)
        self.assertIn("concat_vector == 0 || concat_vector == 1", self.result)
        self.assertIn("packed_score == 0 || packed_score == 1", self.result)
        self.assertIn("num_heads <= 16 && score_stride >= 2048", self.result)
        self.assertIn("head_size % 4 == 0", self.result)

    def test_each_shader_definition_occurs_once(self):
        for name in ("QsaVectorConcatProgram", "QsaPackedScoreProgram",
                     "SparseAttentionIndexerQsaConcatProgram", "SparseAttentionIndexerQsaSelectProgram",
                     "SparseAttentionIndexerCsaCopyCompressedProgram"):
            self.assertEqual(self.result.count(f"Status {name}::GenerateShaderCode("), 1, name)

    def test_private_shaders_retain_the_qualified_math(self):
        self.assertIn('Program{"QsaConcatVector4"}', self.result)
        self.assertIn('Program{"QsaPackedScore4"}', self.result)
        self.assertIn("dot += query_value(row, local_lane, d, inv_rms) * pooled_key[group * 128u + d]",
                      self.result)
        self.assertIn("if (local_idx == 0u) { visible_count_shared", self.result)


if __name__ == "__main__":
    unittest.main()
