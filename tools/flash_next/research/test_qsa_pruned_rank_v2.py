import inspect
import math
import random
import unittest

from build_qsa_pruned_rank_v2 import transform
from run_qsa_pruned_rank_native import main
from test_qsa_pruned_rank import SOURCE, pruned_rank


class QsaPrunedRankV2Tests(unittest.TestCase):
    def test_partial_chunks_ties_and_nan_ranks_remain_exact(self):
        rng = random.Random(127)
        for n in (1, 31, 32, 33, 511, 512, 513, 1031, 8201):
            scores = [rng.randrange(-31, 32) for _ in range(n)]
            scores[-1] = float("nan")
            order = ([0] if math.isnan(scores[0]) else
                     sorted((i for i, s in enumerate(scores) if not math.isnan(s)),
                            key=lambda i: (-scores[i], i)))
            for topk in (0, 1, 64, 512, 1024):
                expected = {candidate: rank for rank, candidate in enumerate(order[:min(topk, n)])}
                candidates = range(n) if n < 2000 else (0, 1, 31, 32, 511, 512, 4095, 8199, 8200)
                for candidate in candidates:
                    self.assertEqual(pruned_rank(scores, candidate, topk, chunk=512),
                                     expected.get(candidate), (n, candidate, topk))

    def test_control_values_and_early_return_are_explicitly_workgroup_uniform(self):
        source = SOURCE.read_text()
        actual = transform(source)
        self.assertIn("workgroupUniformLoad(&rank_header)", actual)
        self.assertIn("workgroupUniformLoad(&rank_rejected)", actual)
        self.assertIn("base += 512u", actual)
        self.assertIn("offset < 512u; offset += 32u", actual)
        self.assertIn("candidate != 0u && current_rank >= selected_blocks", actual)
        self.assertNotIn("candidate != 0u && subgroupAdd(rank) >= selected_blocks", actual)
        self.assertEqual(actual.count("diagnostic(off"), source.count("diagnostic(off"))
        self.assertEqual(inspect.signature(main).parameters["variant"].default, "qsa-pruned-rank-v1")


if __name__ == "__main__":
    unittest.main()
