from pathlib import Path
import unittest

from dense_tile_parallel3_narrow_transform import FLAG, NAME, transform
from dense_tile_parallel3_transform import PROGRAM, transform as broad_transform
from prepare_dense_tile_parallel3_narrow import ROOT, build_source, controller_source


class DenseNarrowTests(unittest.TestCase):
    def test_only_selected_shape_and_private_flag_differ_from_exact_broad_source(self):
        original = Path(
            "/home/sayanshaw/repos/qwen38-int4-dense-columns8-double-buffer/build-v1/matmul_nbits.cc").read_text()
        narrow = transform(original)
        normalized = narrow.replace(FLAG, "ORT_WEBGPU_DENSE_TILE_PARALLEL3").replace(
            "N_ == 2560 && (K_ == 2560 || K_ == 6144)", "(K_ == 2560 || K_ == 6144)")
        self.assertEqual(normalized, broad_transform(original))
        self.assertIn(PROGRAM, narrow)
        self.assertIn(NAME, narrow)

    def test_fresh_build_and_gpu_route_filter_cover_narrow_and_original_wide(self):
        builder = build_source((ROOT / "build_dense_tile_parallel3.py").read_text())
        self.assertIn('"supported_n": 2560', builder)
        self.assertIn("from dense_tile_parallel3_narrow_transform import", builder)
        self.assertIn("qwen38-int4-dense-tile-parallel3-narrow/build-v1", builder)
        source = controller_source((ROOT / "run_dense_tile_parallel3_v1.py").read_text())
        self.assertIn("bool(selected and n == 2560 and k in (2560, 6144))", source)
        self.assertIn('"selected": n == 2560 and k in (2560, 6144)', source)
        self.assertIn('VARIANT = "dense-tile-parallel3-narrow-v1"', source)
        self.assertIn('"full_native_fallback_qualification_required": True', source)
        compile(builder, "narrow-builder", "exec")
        compile(source, "narrow-controller", "exec")


if __name__ == "__main__":
    unittest.main()
