import math
from pathlib import Path
import random
import unittest

from build_qsa_pruned_rank import FLAG, pruned_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qsa-score16/build-v5/sparse_attention_indexer.cc")


def pruned_rank(scores, candidate, topk, chunk=32):
    if candidate == 0 and math.isnan(scores[0]):
        return 0 if topk > 0 else None
    if math.isnan(scores[0]) or math.isnan(scores[candidate]):
        return None
    rank = 0
    for base in range(0, len(scores), chunk):
        rank += sum(value > scores[candidate] or (value == scores[candidate] and other < candidate)
                    for other, value in enumerate(scores[base:base+chunk], base))
        if candidate != 0 and rank >= min(topk, len(scores)):
            return None
    return rank if rank < min(topk, len(scores)) else None


class QsaPrunedRankTests(unittest.TestCase):
    def test_exact_finite_nan_tie_and_partial_subgroup_ranks(self):
        rng = random.Random(127)
        cases = [[rng.randrange(-9, 10) for _ in range(n)] for n in (1, 31, 32, 33, 65, 520, 1031)]
        cases += [[float("nan"), 9, 8], [0.0, -0.0, float("nan"), float("inf"), -float("inf")],
                  [float("nan")] * 33]
        for scores in cases:
            order = ([0] if math.isnan(scores[0]) else
                     sorted((i for i, s in enumerate(scores) if not math.isnan(s)),
                            key=lambda i: (-scores[i], i)))
            for topk in (0, 1, 3, 32, 64, 512, 2048):
                expected = {candidate: rank for rank, candidate in enumerate(order[:min(topk, len(scores))])}
                actual = {candidate: rank for candidate in range(len(scores))
                          if (rank := pruned_rank(scores, candidate, topk)) is not None}
                self.assertEqual(actual, expected, (len(scores), topk))

    def test_candidate_zero_counts_all_valid_scores_and_uses_no_shared_barriers(self):
        private = pruned_program(SOURCE.read_text())
        self.assertIn("candidate != 0u && subgroupAdd(rank) >= selected_blocks", private)
        self.assertIn("candidate == 0u && other_score == other_score", private)
        self.assertIn("final_valid_count", private)
        self.assertNotIn("workgroupBarrier", private)
        self.assertNotIn("var<workgroup>", private)
        self.assertIn("other < block_count", private)
        self.assertIn("other_score == score && other < candidate", private)

    def test_original_class_dispatch_and_hardware_fallback_remain(self):
        source = SOURCE.read_text()
        actual = transform(source)
        start = source.index("class SparseAttentionIndexerQsaParallelRankProgram final")
        stop = source.index("\nStatus CheckShape(", start)
        self.assertIn(source[start:stop], actual)
        start = source.index("    if (use_parallel_rank && phase == QsaCachePhase::Select)")
        stop = source.index("    const uint32_t phase_sort_capacity", start)
        self.assertIn(source[start:stop], actual)
        self.assertIn(f'"{FLAG}", 0', actual)
        self.assertIn("context.HasFeature(wgpu::FeatureName::Subgroups)", actual)
        self.assertIn("subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32", actual)
        self.assertIn(".SetWorkgroupSize(32)", actual)


if __name__ == "__main__":
    unittest.main()
