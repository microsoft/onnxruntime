import hashlib
import json
from pathlib import Path
import unittest

from build_dsa_partition32 import transform as transform_partition32
from build_dsa_query_cache import cached_program, transform


BASE = Path("/home/sayanshaw/repos")
SOURCE = BASE / "qwen38-int4-qmoe-prefill1024/build-v2/dynamic_sparse_attention.cc"
HEADER = BASE / "onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/dynamic_sparse_attention.h"


class QueryCacheTests(unittest.TestCase):
    def test_exact_lane_address_order_and_host_admission(self):
        source, header = SOURCE.read_text(), HEADER.read_text()
        program = cached_program(source, header)
        self.assertIn("for (uint32_t slot = 0; slot < 8; ++slot)", program)
        self.assertIn("dot_product += cached_query", program)
        self.assertIn("slot * 32", program)
        self.assertNotIn("for (var c = local_idx;", program)
        actual = transform(source, header)
        self.assertIn("query_cache_mode == 1 && split_kv && parameters.head_size == 256", actual)
        self.assertIn("constexpr uint32_t split_partition_size = 64;", actual)
        self.assertEqual(actual.count("return context.RunProgram(attention_program);"), 1)
        for lane in range(32):
            self.assertEqual(list(range(lane, 256, 32)), [lane + slot * 32 for slot in range(8)])
        start = source.index("Status DynamicSparseAttentionProgram::GenerateShaderCode")
        stop = source.index("Status DynamicSparseAttentionMergeProgram::GenerateShaderCode", start)
        self.assertIn(source[start:stop], actual)

    def test_extracted_routing_helper_reproduces_frozen_partition32_source(self):
        component = BASE / "qwen38-int4-qmoe-prefill512/build-v3"
        manifest = json.loads((BASE / "qwen38-int4-dsa-partition32/build-v1/build-manifest.json").read_text())
        actual = transform_partition32((component / "dynamic_sparse_attention.cc").read_text(), HEADER.read_text())
        self.assertEqual(hashlib.sha256(actual.encode()).hexdigest(),
                         manifest["source_sha256"]["dynamic_sparse_attention.cc"])

    def test_v2_preserves_fp32_fallback_without_changing_shader(self):
        source, header = SOURCE.read_text(), HEADER.read_text()
        original = transform(source, header)
        manifest = json.loads((BASE / "qwen38-int4-dsa-query-cache/build-v1/build-manifest.json").read_text())
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),
                         manifest["source_sha256"]["dynamic_sparse_attention.cc"])
        candidate = transform(source, header, fp16_only=True)
        self.assertEqual(candidate, original.replace(
            "query_cache_mode == 1 && split_kv && parameters.head_size == 256;",
            "query_cache_mode == 1 && split_kv && parameters.head_size == 256 && query->IsDataType<MLFloat16>();"))


if __name__ == "__main__":
    unittest.main()
