import inspect
import math
import random
import unittest

from build_qsa_subgroup_rank import FLAG, subgroup_program, transform
from run_qsa_pruned_rank_native import main
from test_qsa_pruned_rank import SOURCE


class QsaSubgroupRankTests(unittest.TestCase):
    def test_integer_reduction_matches_all_pairs_and_valid_count(self):
        rng = random.Random(127)
        for n in (1, 31, 32, 33, 63, 64, 65, 520, 8201):
            scores = [rng.randrange(-9, 10) for _ in range(n)]
            scores[-1] = float("nan")
            for candidate in {0, min(32, n-1), n-1}:
                ranks = [sum(scores[i] > scores[candidate] or (scores[i] == scores[candidate] and i < candidate)
                             for i in range(lane, n, 64)) for lane in range(64)]
                valid = [sum(not math.isnan(scores[i]) for i in range(lane, n, 64)) for lane in range(64)]
                self.assertEqual(sum(ranks[:32]) + sum(ranks[32:]),
                                 sum(s > scores[candidate] or (s == scores[candidate] and i < candidate)
                                     for i, s in enumerate(scores)))
                self.assertEqual(sum(valid[:32]) + sum(valid[32:]), sum(not math.isnan(s) for s in scores))

    def test_original_thread_ownership_output_semantics_and_single_barrier(self):
        source = SOURCE.read_text()
        private = subgroup_program(source)
        self.assertIn("other += 64u", private)
        self.assertEqual(private.count("workgroupBarrier()"), 1)
        self.assertIn("if (sg_id == 0u)", private)
        self.assertIn("ranks[subgroup_index] = subgroup_rank", private)
        self.assertIn("rank = ranks[0] + ranks[1]", private)
        self.assertIn("min(selected_blocks, (valid_counts[0] + valid_counts[1]))", private)
        self.assertNotIn("rank_rejected", private)
        actual = transform(source)
        self.assertIn(f'"{FLAG}", 0', actual)
        self.assertIn("context.HasFeature(wgpu::FeatureName::Subgroups)", actual)
        self.assertIn("subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32", actual)
        self.assertEqual(actual.count("diagnostic(off"), source.count("diagnostic(off"))
        start = source.index("class SparseAttentionIndexerQsaParallelRankProgram final")
        stop = source.index("\nStatus CheckShape(", start)
        self.assertIn(source[start:stop], actual)
        self.assertEqual(inspect.signature(main).parameters["selector"].default, "ORT_WEBGPU_QSA_PRUNED_RANK")


if __name__ == "__main__":
    unittest.main()
