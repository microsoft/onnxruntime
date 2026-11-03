from pathlib import Path
import random
import unittest

from build_qsa_parallel_map import SELECTOR, transform


def emulate(mask):
    chunks = [mask[i:i + 512] for i in range(0, len(mask), 512)]
    counts = [sum(v != 0 for v in chunk) for chunk in chunks]
    output = [None] * (len(mask) + 1)
    writers = [0] * len(output)
    for chunk, values in enumerate(chunks):
        offset = sum(counts[:chunk])
        for token, value in enumerate(values):
            if value != 0:
                output[offset] = chunk * 512 + token
                writers[offset] += 1
                offset += 1
        if chunk == len(chunks) - 1:
            output[-1] = offset
            writers[-1] += 1
    return output, writers


class ParallelMapTests(unittest.TestCase):
    def test_exact_stable_compaction_and_disjoint_writes(self):
        rng = random.Random(1903)
        for length in (1, 63, 64, 65, 511, 512, 513, 8191, 8192, 8193, 32768, 33024, 65536):
            masks = [[1] * length, [0] * length,
                     [rng.choice((0, -1, 1 << 40, -(1 << 63))) for _ in range(length)],
                     [int(i % 512 == 0 or i % 512 == 511) for i in range(length)]]
            for mask in masks:
                output, writers = emulate(mask)
                expected = [i for i, value in enumerate(mask) if value != 0]
                self.assertEqual(output[:len(expected)], expected)
                self.assertEqual(output[-1], len(expected))
                self.assertEqual(writers, [1] * len(expected) +
                                 [0] * (length - len(expected)) + [1])

    def test_rewrite_is_scoped_and_selector_retains_dense_map_gates(self):
        path = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/sparse_attention_indexer.cc")
        original = path.read_text()
        changed = transform(original)
        self.assertIn("parallel_map == 1 && use_dense_map && sequence_length == 1 && total_length >= 8192", SELECTOR)
        self.assertIn("parallel_map == 0 || parallel_map == 1", SELECTOR)
        self.assertEqual(changed.count("if (use_parallel_map && phase == QsaCachePhase::Map) continue;"), 1)
        self.assertIn("if (use_parallel_rank && phase == QsaCachePhase::Select)", changed)


if __name__ == "__main__":
    unittest.main()
