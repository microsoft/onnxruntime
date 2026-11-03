import math
import random
import unittest


class ParallelRankInitializationTests(unittest.TestCase):
    def check_partition(self, scores, ratio, budget, tail):
        blocks = len(scores)
        selected_blocks = min(budget // ratio, blocks)
        capacity = budget + ratio - 1
        visible = [3 * i + 1 for i in range(blocks * ratio + tail)]
        first_nan = bool(scores) and math.isnan(scores[0])
        valid_count = sum(not math.isnan(score) for score in scores)
        valid_selected = min(selected_blocks, 1 if first_nan else valid_count)
        writes = []

        for slot in range(valid_selected * ratio, selected_blocks * ratio):
            writes.append((slot, -1))
        for slot in range(selected_blocks * ratio + tail, capacity):
            writes.append((slot, -1))
        for i in range(tail):
            writes.append((selected_blocks * ratio + i, visible[blocks * ratio + i]))
        for candidate, score in enumerate(scores):
            if first_nan:
                if candidate != 0:
                    continue
                rank = 0
            else:
                if math.isnan(score):
                    continue
                rank = sum(other > score or (other == score and index < candidate)
                           for index, other in enumerate(scores))
            if rank < selected_blocks:
                for i in range(ratio):
                    writes.append((rank * ratio + i, visible[candidate * ratio + i]))

        offsets = [slot for slot, _ in writes]
        self.assertEqual(len(offsets), capacity)
        self.assertEqual(set(offsets), set(range(capacity)), "Missing or overlapping output writes")
        actual = [None] * capacity
        for slot, value in writes:
            actual[slot] = value

        if first_nan:
            order = [0][:selected_blocks]
        else:
            order = sorted(
                (i for i, score in enumerate(scores) if not math.isnan(score)),
                key=lambda i: (-scores[i], i),
            )[:selected_blocks]
        expected = [-1] * capacity
        for rank, block in enumerate(order):
            expected[rank * ratio:(rank + 1) * ratio] = visible[block * ratio:(block + 1) * ratio]
        expected[selected_blocks * ratio:selected_blocks * ratio + tail] = visible[blocks * ratio:]
        self.assertEqual(actual, expected)

    def test_finite_nonfinite_ties_and_empty_partitions(self):
        cases = [
            [], [0.0], [0.0] * 23, [-0.0, 0.0, -0.0, 0.0],
            [math.nan], [math.nan] * 23, [math.nan, 1.0, math.inf],
            [1.0, math.nan, 2.0, math.nan, 0.0],
            [math.inf, math.nan, math.inf, -math.inf, -math.inf],
            [-math.inf, math.nan, -math.inf, -math.inf],
        ]
        for scores in cases:
            for ratio in (1, 2, 4, 7):
                for budget in (ratio, ratio + 1, 3 * ratio, 8 * ratio + 1):
                    for tail in range(ratio):
                        with self.subTest(scores=scores, ratio=ratio, budget=budget, tail=tail):
                            self.check_partition(scores, ratio, budget, tail)

    def test_randomized_disjoint_initialization_and_scatter(self):
        rng = random.Random(20261023)
        values = [-math.inf, -1.0, -0.0, 0.0, 1.0, math.inf, math.nan]
        for case in range(500):
            ratio = rng.randrange(1, 9)
            scores = rng.choices(values, k=rng.randrange(65))
            budget = rng.randrange(ratio, 100)
            tail = rng.randrange(ratio)
            with self.subTest(case=case):
                self.check_partition(scores, ratio, budget, tail)


if __name__ == "__main__":
    unittest.main()
