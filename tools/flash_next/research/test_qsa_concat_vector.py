from pathlib import Path
import unittest

from build_qsa_concat_vector import FLAG, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qsa-parallel-map/build-v1/sparse_attention_indexer.cc")


class ConcatVectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = SOURCE.read_text()
        cls.result = transform(cls.original)

    def test_default_off_and_invalid_selector(self):
        self.assertIn(f'("{FLAG}", 0)', self.result)
        self.assertIn("concat_vector == 0 || concat_vector == 1", self.result)
        self.assertIn("head_size % 4 == 0", self.result)
        self.assertIn('std::string_view{"nvidia"}', self.result)

    def test_fallback_is_frozen(self):
        start = self.original.index("  if (present_elements > 0) {",
                                    self.original.index("Status SparseAttentionIndexer::ComputeQsa"))
        stop = self.original.index("  const int64_t rows = batch_size * sequence_length;", start)
        self.assertIn("  } else {\n" + self.original[start:stop] + "  }\n", self.result)

    def test_all_vector_units_and_types(self):
        self.assertIn('Program{"QsaConcatVector4"}', self.result)
        start = self.result.index("Status QsaVectorConcatProgram::GenerateShaderCode")
        stop = self.result.index("Status SparseAttentionIndexerQsaConcatProgram::GenerateShaderCode", start)
        shader = self.result[start:stop]
        self.assertIn("ShaderUsage::UseValueTypeAlias", shader)
        self.assertNotIn("present_key_element_t", shader)
        for expression in ("present_elements / 4", "head_size / 4",
                           "(packed_qk ? query_shape[2] : head_size) / 4",
                           "(packed_qk ? num_heads * head_size : 0) / 4"):
            self.assertIn("ToUint32(" + expression + ")", self.result)
        for tensor in ("past_key", "current_key"):
            self.assertIn(f"{{{tensor}->Shape().Size() / 4}}, 4", self.result)

    def test_copy_addresses_match_scalar_for_packed_and_batched(self):
        for batch_size in (1, 3):
            for past in (0, 3, 32768):
                for rows in (0, 1, 16):
                    for head in (4, 128):
                        for heads in (0, 5):
                            stride = (heads + 1) * head
                            offset = heads * head
                            total = past + rows
                            for batch in range(batch_size):
                                for token in {0, past - 1, past, total - 1}:
                                    if not 0 <= token < total:
                                        continue
                                    for d in (0, head - 4):
                                        g = (batch * total + token) * (head // 4) + d // 4
                                        self.assertEqual(g * 4, (batch * total + token) * head + d)
                                        if token < past:
                                            address = (batch * past + token) * (head // 4) + d // 4
                                            expected = (batch * past + token) * head + d
                                        else:
                                            row = batch * rows + token - past
                                            address = row * (stride // 4) + offset // 4 + d // 4
                                            expected = row * stride + offset + d
                                        self.assertEqual(address * 4, expected)


if __name__ == "__main__":
    unittest.main()
