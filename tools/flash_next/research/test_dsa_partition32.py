from pathlib import Path
import unittest

from build_dsa_partition32 import private_program, transform


BASE = Path("/home/sayanshaw/repos")
SOURCE = BASE / "qwen38-int4-qmoe-prefill512/build-v3/dynamic_sparse_attention.cc"
HEADER = BASE / "onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/dynamic_sparse_attention.h"


class Partition32Tests(unittest.TestCase):
    def test_shader_clone_is_identical_apart_from_identity(self):
        source, header = SOURCE.read_text(), HEADER.read_text()
        clone = private_program(source, header).replace(
            "DynamicSparseAttentionPartition32Program", "DynamicSparseAttentionProgram").replace(
            'Program{"DynamicSparseAttentionPartition32"}', 'Program{"DynamicSparseAttention"}')
        declaration, method = clone.split("\nStatus DynamicSparseAttentionProgram::", 1)
        self.assertIn(declaration.rstrip(), header)
        self.assertIn("Status DynamicSparseAttentionProgram::" + method, source)

    def test_split_decode_and_dispatch_budget_required_with_original_fallback(self):
        actual = transform(SOURCE.read_text(), HEADER.read_text())
        self.assertIn("partition32_mode == 1 && split_kv &&", actual)
        self.assertIn("((candidates_limit + 31u) / 32u)", actual)
        self.assertIn("partition32 ? 32u : 64u", actual)
        self.assertIn("partition32_mode == 0 || partition32_mode == 1", actual)
        self.assertEqual(actual.count("return context.RunProgram(attention_program);"), 1)
        self.assertEqual(actual.count("ORT_RETURN_IF_ERROR(run_attention(attention_program));"), 2)
        self.assertIn("DynamicSparseAttentionProgram attention_program(", actual)
        self.assertIn("DynamicSparseAttentionPartition32Program attention_program(", actual)
        original_merge = SOURCE.read_text().split("  if (partitions > 1) {\n    DynamicSparseAttentionMergeProgram", 1)[1]
        self.assertTrue(actual.endswith("  if (partitions > 1) {\n    DynamicSparseAttentionMergeProgram" + original_merge))


if __name__ == "__main__":
    unittest.main()
