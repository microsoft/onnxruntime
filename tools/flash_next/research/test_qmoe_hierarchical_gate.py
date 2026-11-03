from pathlib import Path
import unittest

import numpy as np

from build_qmoe_hierarchical_gate import FLAG, transform


def hierarchical_topk(values, size):
    groups = [sorted(range(start, start + size), key=lambda index: (-float(values[index]), index))[:10]
              for start in range(0, len(values), size)]
    cursors = [0] * len(groups)
    result = []
    for _ in range(10):
        group = min((i for i in range(len(groups)) if cursors[i] < 10),
                    key=lambda i: (-float(values[groups[i][cursors[i]]]), groups[i][cursors[i]]))
        result.append(groups[group][cursors[group]])
        cursors[group] += 1
    return result


class HierarchicalGateTests(unittest.TestCase):
    def test_random_ties_signed_zeros_and_negative_infinity(self):
        rng = np.random.default_rng(20261029)
        cases = [np.zeros(512, dtype=np.float16),
                 np.tile(np.array([0.0, -0.0], dtype=np.float16), 256),
                 np.full(512, -np.inf, dtype=np.float16)]
        cases[-1][511] = 1.0
        cases += [rng.integers(-8, 8, size=512).astype(np.float16) for _ in range(32)]
        for values in cases:
            expected = sorted(range(512), key=lambda index: (-float(values[index]), index))[:10]
            for size in (32, 64, 128):
                with self.subTest(size=size):
                    self.assertEqual(hierarchical_topk(values, size), expected)

    def test_all_winners_can_come_from_one_subgroup(self):
        for size in (32, 64, 128):
            for start in (0, 512 - size):
                values = np.zeros(512, dtype=np.float16)
                values[start:start + 10] = np.arange(10, 0, -1, dtype=np.float16)
                self.assertEqual(hierarchical_topk(values, size), list(range(start, start + 10)))
                self.assertLessEqual((512 // size) * 10, 160)
                self.assertLessEqual(512 // size, size)

    def test_selector_is_validated_cached_and_capability_gated(self):
        path = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-fc2-final-mix/build-v2/overlay/contrib_ops/webgpu/moe/qmoe.cc")
        source = transform(path.read_text())
        self.assertIn(f'"{FLAG} must be 0 or 1"', source)
        self.assertIn("key.push_back(hierarchical_gate)", source)
        self.assertIn("context.AdapterInfo().subgroupMinSize >= 32", source)
        self.assertIn("const bool use_hierarchical = use_subgroups &&", source)
        self.assertIn("QmoeHierarchicalGateProgram gate;", source)
        self.assertIn("Gate1TokenProgram gate{k_", source)


if __name__ == "__main__":
    unittest.main()
