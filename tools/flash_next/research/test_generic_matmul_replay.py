from pathlib import Path
import unittest

from build_generic_matmul_replay import SELECTOR, transform
from prepare_generic_matmul_replay import SHAPES


class GenericMatMulReplayTests(unittest.TestCase):
    def test_expansion_does_not_change_any_shader_or_existing_exact_route(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-static-node-replay/build-v8/overlay/contrib_ops/webgpu/quantization/matmul_nbits.cc").read_text()
        changed = transform(source)
        self.assertEqual(changed.replace(SELECTOR, "  if (!exact_route) return compute();"), source)
        self.assertIn("generic_replay == 0 || generic_replay == 1", SELECTOR)
        for flag in ("ORT_WEBGPU_MATMUL_NBITS_SKINNY", "ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY"):
            self.assertIn(f'("{flag}", 0) == 0', SELECTOR)

    def test_generic_shape_scope_does_not_overlap_switchable_exact_shapes(self):
        self.assertEqual(len(SHAPES), 9)
        self.assertTrue(set(SHAPES).isdisjoint({(2560, 512), (2560, 640), (10240, 324)}))


if __name__ == "__main__":
    unittest.main()
