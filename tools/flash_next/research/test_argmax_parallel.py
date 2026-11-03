import math
from pathlib import Path
import unittest

from build_argmax_parallel import FLAG, PROGRAM, transform


def original(values):
    best, index = values[0], 0
    for i, value in enumerate(values[1:], 1):
        if value > best:
            best, index = value, i
    return index


def parallel(values):
    partials = []
    for start in range(0, len(values), 2048):
        lanes = []
        for lane in range(128):
            best, index = values[0], 0
            for i in range(start + lane, min(start + 2048, len(values)), 128):
                if values[i] > best:
                    best, index = values[i], i
            lanes.append((best, index))
        for stride in (64, 32, 16, 8, 4, 2, 1):
            for lane in range(stride):
                value, index = lanes[lane + stride]
                best, best_index = lanes[lane]
                if value > best or (value == best and index < best_index):
                    lanes[lane] = value, index
        partials.append(lanes[0])
    best, index = partials[0]
    for value, candidate_index in partials:
        if value > best or (value == best and candidate_index < index):
            best, index = value, candidate_index
    return index


class ArgMaxParallelTests(unittest.TestCase):
    def test_ordered_nan_tie_zero_and_tail_cases(self):
        for n in (1, 17, 31, 32, 127, 128, 129, 2047, 2048, 2049, 4097, 248320):
            base = [math.sin(i * 0.071) for i in range(n)]
            cases = [base, [0.0] * n, [-0.0] * n, [-math.inf] * n,
                     [math.nan] + base[1:], base[:-1] + [math.nan],
                     [-1.0] * (n - 1) + [math.inf],
                     [0.0] * (n - 1) + [math.ulp(0.0)]]
            tied = base.copy()
            tied[min(n - 1, 127)] = tied[n - 1] = math.inf
            cases.append(tied)
            for case in cases:
                with self.subTest(n=n, expected=original(case)):
                    self.assertEqual(parallel(case), original(case))

    def test_default_branch_preserved(self):
        source = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/onnxruntime/core/providers/webgpu/reduction/reduction_ops.cc").read_text()
        candidate = transform(source)
        suffix = source[source.index("  if (use_naive_reduction) {"):]
        self.assertTrue(candidate.endswith(suffix))
        self.assertIn(f'{FLAG} must be 0 or 1', candidate)
        self.assertIn("select_last_index_ == 0", candidate)
        self.assertIn("input_axes.size() == 1 && are_axes_innermost", candidate)
        self.assertEqual(candidate.count("return context.RunProgram(finish);"), 1)

    def test_uniform_barrier_and_unique_partial_ownership(self):
        self.assertEqual(PROGRAM.count("workgroupBarrier();"), 4)
        self.assertIn("i += 128u", PROGRAM)
        self.assertIn('ix.SetByOffset("workgroup_idx"', PROGRAM)
        self.assertNotIn("atomic", PROGRAM)
        self.assertIn("var index = 0u", PROGRAM)


if __name__ == "__main__":
    unittest.main()
