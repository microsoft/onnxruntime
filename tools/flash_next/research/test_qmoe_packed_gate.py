from pathlib import Path
import unittest

import numpy as np

from build_qmoe_packed_gate import transform


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-expert-bitcast-unpack/build-v2/overlay/contrib_ops/webgpu/moe/qmoe.cc")


def keys(values):
    with np.errstate(invalid="ignore"):
        floats = values.astype(np.float32)
        bits = np.where(floats == 0, np.uint32(0), floats.view(np.uint32))
    return np.where(bits & np.uint32(0x80000000),
                    ~bits, bits ^ np.uint32(0x80000000)).astype(np.uint32)


class PackedGateTests(unittest.TestCase):
    def test_compression_preserves_all_half_ordering_and_expert_ties(self):
        half = np.arange(65536, dtype=np.uint16).view(np.float16)
        ordered = keys(half)
        compressed = ordered >> 13
        indices = np.argsort(compressed, kind="stable")
        adjacent_bucket_equal = compressed[indices[1:]] == compressed[indices[:-1]]
        self.assertTrue(np.all(ordered[indices[1:]][adjacent_bucket_equal] ==
                               ordered[indices[:-1]][adjacent_bucket_equal]))
        ids = np.tile(np.array([0, 1, 511], dtype=np.uint32), len(half))
        ordered = np.repeat(ordered, 3)
        packed = (((ordered >> 13) + 1) << 9) | (511 - ids)
        self.assertGreater(int(packed.min()), 0)
        self.assertLess(int(packed.max()), 2**29)
        expected = np.lexsort((ids, -ordered.astype(np.int64)))
        actual = np.argsort(-packed.astype(np.int64), kind="stable")
        np.testing.assert_array_equal(expected, actual)

    def test_subgroup_reductions_and_merge_match_original_top10(self):
        rng = np.random.default_rng(20261029)
        patterns = [np.zeros(512, dtype=np.float16),
                    np.array([0.0, -0.0] * 256, dtype=np.float16)]
        for _ in range(24):
            patterns.append(rng.integers(0, 65536, size=512, dtype=np.uint16).view(np.float16))
        for size in (32, 64, 128):
            for values in patterns:
                ordered = keys(values)
                ids = np.arange(512, dtype=np.uint32)
                packed = (((ordered >> 13) + 1) << 9) | (511 - ids)
                rows = packed.reshape(-1, size).copy()
                candidates = []
                for _ in range(10):
                    best = np.max(rows, axis=1)
                    candidates.append(best)
                    rows = np.where(rows == best[:, None], np.uint32(0), rows)
                rows = np.stack(candidates, axis=1)
                np.testing.assert_array_equal(rows, np.sort(packed.reshape(-1, size), axis=1)[:, ::-1][:, :10])
                cursors = [0] * len(rows)
                actual = []
                for _ in range(10):
                    heads = [int(row[cursor]) if cursor < 10 else 0 for row, cursor in zip(rows, cursors)]
                    winner = int(np.argmax(heads))
                    actual.append(511 - (heads[winner] & 511))
                    cursors[winner] += 1
                np.testing.assert_array_equal(actual, np.lexsort((ids, -ordered.astype(np.int64)))[:10])

    def test_selector_validation_key_and_original_fallback(self):
        result = transform(SOURCE.read_text())
        self.assertEqual(result.count("ORT_WEBGPU_QMOE_PACKED_GATE must be 0 or 1"), 2)
        self.assertIn("key.push_back(packed_gate);", result)
        self.assertIn("if (use_hierarchical && packed_gate == 1 && router_logits->IsDataType<MLFloat16>())", result)
        self.assertIn("QmoeHierarchicalGateProgram gate;", result)
        shader = (ROOT / "qmoe_packed_gate.inc").read_text()
        self.assertIn("let best_key = subgroupMax(key);", shader)
        self.assertNotIn("subgroupShuffleXor", shader)
        self.assertNotIn("subgroupMin", shader)
        self.assertIn("sum += exp(f32(selected_vals[rank]) - max_val)", shader)


if __name__ == "__main__":
    unittest.main()
