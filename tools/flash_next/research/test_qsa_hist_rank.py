import math
from pathlib import Path
import random
import struct
import unittest

from build_qsa_hist_rank import BUCKET, FLAG, bounded_program, transform
from run_qsa_pruned_rank_native import rank_stride_eligible


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def bucket(score, maximum):
    if maximum == 0 or not math.isfinite(maximum):
        return 0
    return int(min(255, max(0, f32(f32(score / maximum) * 255))))


class HistogramRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qsa-prefill-score16/"
                          "build-v1/sparse_attention_indexer.cc").read_text()

    def test_bound_never_excludes_a_selected_candidate(self):
        rng = random.Random(3101)
        cases = [[f32(rng.random() * 100) for _ in range(8192)],
                 [0.0] * 4096, [1.0] * 4096, [-0.0, 0.0, float("inf")] * 1024,
                 [float("nan")] * 4096, [-1.0, 1.0, 2.0] * 1024]
        for scores in cases:
            valid = [v for v in scores if not math.isnan(v)]
            maximum = float("inf") if any(v < 0 for v in valid) else max(valid, default=0)
            bins = [0] * 256
            for score in valid:
                bins[bucket(score, maximum)] += 1
            count, cutoff = 0, 0
            for b in range(255, -1, -1):
                count += bins[b]
                if count >= 512:
                    cutoff = b
                    break
            selected = sorted((i for i, v in enumerate(scores) if not math.isnan(v)),
                              key=lambda i: (-scores[i], i))[:512]
            self.assertTrue(all(bucket(scores[i], maximum) >= cutoff for i in selected))

    def test_original_comparisons_and_barriers_remain(self):
        program = bounded_program(self.source)
        self.assertIn("other_score > score || (other_score == score && other < candidate)", program)
        self.assertIn("candidate == 0u && other_score == other_score", program)
        self.assertIn("first_score != first_score", program)
        self.assertIn("candidate != 0u && candidate < block_count", program)
        start = self.source.index("class SparseAttentionIndexerQsaParallelRankProgram final")
        stop = self.source.index("\nStatus CheckShape(", start)
        self.assertEqual(program.count("workgroupBarrier();"), self.source[start:stop].count("workgroupBarrier();"))
        self.assertGreater(program.index("if (pruned) { return; }"), program.rindex("workgroupBarrier();"))
        result = transform(self.source)
        self.assertIn(self.source[start:stop], result)
        self.assertIn("score_stride >= 2048", result)
        self.assertIn(f"{FLAG} must be 0 or 1", result)
        self.assertEqual(result.count(BUCKET), 2)
        self.assertIn("DataTypeImpl::GetType<uint32_t>(), TensorShape({2})", result)

    def test_window_and_long_only_gates_are_explicit(self):
        root = Path(__file__).parent
        text = (root / "benchmark_webgpu_int4.py").read_text()
        for marker in ("--compare-qsa-hist-rank-windows", "args.compare_qsa_hist_rank_windows or",
                       "QSA_HIST_RANK_FLAG if args.compare_qsa_hist_rank_windows", f'"{FLAG}",'):
            self.assertIn(marker, text)
        self.assertIn('QSA_HIST_RANK_FLAG: "qsa_hist_rank"', (root / "qsa_rank_decode_windows.py").read_text())
        runner = (root / "run_hc_up_parallel2_model.py").read_text()
        self.assertIn("initial_context=2048", runner)
        self.assertIn("windows{initial_context}", runner)
        self.assertIn("profile_context = args.context if histogram_qualification else initial_context", runner)
        self.assertIn("profile{profile_context}", runner)
        self.assertIn("Long-only standard qualification is not yet available", runner)
        self.assertIn("initial_context=8192", (root / "run_qsa_hist_rank_model.py").read_text())

    def test_native_admission_requires_actual_original_score_stride(self):
        for stride in (512, 2047, 2048, 8192):
            events = [{"cat": "Kernel", "name": "node&SparseAttentionIndexerQsaParallelRank",
                       "args": {"shapes": f"inputs[1] = {{1,{stride}}}"}}]
            self.assertEqual(rank_stride_eligible(events, 2048), stride >= 2048)
        self.assertFalse(rank_stride_eligible([], 2048))

    def test_uniform_exit_is_workgroup_synchronized_and_preserves_original_rank(self):
        program = bounded_program(self.source, uniform_exit=True)
        self.assertIn("workgroupUniformLoad(&pruning)", program)
        self.assertLess(program.index("if (pruned) { return; }"), program.index("var rank = 0u"))
        self.assertIn("if (local_idx == 0u) { pruning = select(0u, 1u, should_prune); }", program)
        self.assertIn("candidate != 0u && candidate < block_count", program)
        self.assertIn("other_score > score || (other_score == score && other < candidate)", program)
        result = transform(self.source, minimum_stride=8192, uniform_exit=True)
        self.assertIn("score_stride >= 8192", result)
        self.assertIn("QsaHistogramUniformRankProgram rank;", result)
        self.assertIn('Program{"QsaHistogramUniformRank"}', result)

    def test_original_v1_transform_is_byte_identical_to_built_source(self):
        previous = Path("/home/sayanshaw/repos/qwen38-int4-qsa-hist-rank/"
                        "build-v1/sparse_attention_indexer.cc").read_text()
        self.assertEqual(transform(self.source), previous)


if __name__ == "__main__":
    unittest.main()
