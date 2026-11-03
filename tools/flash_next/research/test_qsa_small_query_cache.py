from pathlib import Path
import unittest

from build_qsa_small_query_cache import FLAG, PROGRAM, STORAGE_BYTES, cached_program, transform
from run_qsa_small_query_cache_native import eligible


class SmallQueryCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qsa-prefill-score16/"
                          "build-v1/sparse_attention_indexer.cc").read_text()

    def test_original_prefill_and_decode_programs_and_dispatch_preserved(self):
        result = transform(self.source)
        start = self.source.index("class QsaPackedScore16Program final")
        function = self.source.index("Status QsaPackedScore16Program::GenerateShaderCode", start)
        stop = self.source.index("\nclass QsaPackedScoreProgram final", function)
        self.assertIn(self.source[start:stop], result)
        start = self.source.index("    if (score16 == 1 && (packed_score == 1 || indexer_stack == 1)")
        stop = self.source.index("    if ((packed_score == 1 || indexer_stack == 1)", start + 1)
        self.assertIn(self.source[start:stop], result)
        self.assertIn('CacheHint(query->GetElementType(), num_heads, head_size', result)
        self.assertIn('"prefill-score16-2d"', result)
        self.assertEqual(result.count("class QsaPackedScoreProgram final"), 1)
        self.assertEqual(result.count("class QsaPackedScore16Program final"), 1)
        self.assertEqual(result.count("class QsaSmallQueryCacheProgram final"), 1)
        start = self.source.index("    if (prefill_score16 == 1")
        stop = self.source.index("    if (score16 == 1", start)
        self.assertIn(self.source[start:stop], result)

    def test_only_candidate_count_storage_and_query_load_stride_change(self):
        original_start = self.source.index("class QsaPackedScore16Program final")
        original_function = self.source.index("Status QsaPackedScore16Program::GenerateShaderCode", original_start)
        original_stop = self.source.index("\nclass QsaPackedScoreProgram final", original_function)
        original = self.source[original_start:original_stop]
        result = cached_program(self.source)
        for marker in ("dot +=", "square_sum +=", "query_value(", "pooled_value(", "group * 144u", "local_idx / 16u"):
            self.assertEqual([line for line in original.splitlines() if marker in line],
                             [line for line in result.splitlines() if marker in line])
        self.assertEqual(STORAGE_BYTES, 4468)
        self.assertIn(f'Program{{"{PROGRAM}"}}', result)
        self.assertIn("head_scores: array<f32, 16>", result)

    def test_candidate_ownership_query_loads_and_tails(self):
        for stride in (128, 129, 511, 512, 513, 2047):
            writers = [group * 4 + thread // 16 for group in range((stride + 3) // 4)
                       for thread in range(64) if thread % 16 == 0 and group * 4 + thread // 16 < stride]
            self.assertEqual(writers, list(range(stride)))
        loads = [offset for thread in range(64) for offset in range(thread, 512, 64)]
        self.assertEqual(sorted(loads), list(range(512)))

    def test_benchmark_and_runner_wiring(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-qsa-small-query-cache-windows", benchmark)
        self.assertIn("args.compare_qsa_small_query_cache_windows or", benchmark)
        self.assertIn('"qsa_small_query_cache_decode_windows" if args.compare_qsa_small_query_cache_windows', benchmark)
        self.assertIn("QSA_SMALL_QUERY_CACHE_FLAG if args.compare_qsa_small_query_cache_windows", benchmark)
        self.assertIn(f'"{FLAG}",', benchmark)
        windows = (root / "qsa_rank_decode_windows.py").read_text()
        self.assertIn('QSA_SMALL_QUERY_CACHE_FLAG: "qsa_small_query_cache"', windows)
        runner = (root / "run_hc_up_parallel2_model.py").read_text()
        self.assertIn('variant == "qsa-small-query-cache-v2"', runner)

    def test_small_stride_and_workgroup_limits_and_invalid_selector(self):
        result = transform(self.source)
        self.assertIn("small_query_cache == 0 || small_query_cache == 1", result)
        self.assertIn(f"{FLAG} must be 0 or 1", result)
        self.assertIn("score_stride >= 128 && score_stride < 2048", result)
        self.assertIn("maxComputeWorkgroupStorageSize >= 4468", result)
        self.assertIn("maxComputeWorkgroupSizeX >= 64", result)
        self.assertIn("maxComputeInvocationsPerWorkgroup >= 64", result)
        self.assertIn("QsaSmallQueryCacheProgram select", result)

    def test_native_admission_boundaries(self):
        def events(stride, heads=4, rows=1, name="SparseAttentionIndexerQsaScore"):
            return [dict(cat="Kernel", name="fixture&" + name,
                         args=dict(shapes=f"inputs[0] = {{1,{rows},640}} outputs[2] = {{1,{stride}}}",
                                   cache_key=f"QsaPackedScore4[10|{heads}|128]")) for _ in range(18)]
        for stride in (128, 512, 1024, 2047):
            self.assertTrue(eligible(events(stride)))
        for stride in (0, 127, 2048, 8192, 16384):
            self.assertFalse(eligible(events(stride)))
        self.assertFalse(eligible(events(512, heads=8)))
        self.assertFalse(eligible(events(512, rows=2)))
        self.assertFalse(eligible([]))
        self.assertTrue(eligible(events(512, name="QsaPackedScore4")))
        nonglobal = events(512)
        for event in nonglobal:
            event["args"]["shapes"] = "inputs[0] = {1,1,640} outputs[0] = {1,1,2048}"
        self.assertFalse(eligible(nonglobal))


if __name__ == "__main__":
    unittest.main()
