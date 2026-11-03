from pathlib import Path
import unittest

from dense_tile_parallel3_long_k_transform import FLAG, transform
from dense_tile_parallel3_narrow_transform import transform as narrow_transform
from dense_tile_parallel3_transform import PROGRAM
from prepare_dense_tile_parallel3_long_k import ROOT, build_source


class DenseLongKTests(unittest.TestCase):
    def test_only_long_k_admission_and_flag_change_the_already_measured_shader(self):
        original = Path(
            "/home/sayanshaw/repos/qwen38-int4-dense-columns8-double-buffer/build-v1/matmul_nbits.cc").read_text()
        long_k = transform(original)
        normalized = long_k.replace(FLAG, "ORT_WEBGPU_DENSE_TILE_PARALLEL3_NARROW").replace(
            "N_ == 2560 && K_ == 6144 &&",
            "N_ == 2560 && (K_ == 2560 || K_ == 6144) &&")
        self.assertEqual(normalized, narrow_transform(original))
        self.assertIn(PROGRAM, long_k)
        source = build_source((ROOT / "build_dense_tile_parallel3.py").read_text())
        self.assertIn('"supported_k": [6144], "supported_n": 2560', source)
        self.assertIn("from dense_tile_parallel3_long_k_transform import", source)
        compile(source, "long-k-builder", "exec")


if __name__ == "__main__":
    unittest.main()
