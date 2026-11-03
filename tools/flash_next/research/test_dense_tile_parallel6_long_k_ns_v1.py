from pathlib import Path
import unittest

from dense_tile_parallel3_transform import PROGRAM as THREE_PROGRAM
from dense_tile_parallel6_long_k_transform import NAME, PROGRAM, SHARED_BYTES, transform
from prepare_dense_tile_parallel6_long_k_ns_v1 import builder_source, controller_source


ROOT = Path(__file__).parent


class SixWorkerTests(unittest.TestCase):
    def test_exact_original_gpu_body_except_parallel_geometry(self):
        normalized = PROGRAM.replace(NAME, "DenseTileParallel3Packed")
        for before, after in (
            ("tile_a: array<vec4<f16>, 3072>", "tile_a: array<vec4<f16>, 1536>"),
            ("wave * 192u", "wave * 96u"),
            ("wave * 6u + tile_worker", "wave * 3u + tile_worker"),
            ("(wave % 2u) * 6u + tile_worker", "(wave % 2u) * 3u + tile_worker"),
        ):
            normalized = normalized.replace(before, after)
        self.assertEqual(normalized, THREE_PROGRAM)
        self.assertEqual(SHARED_BYTES, 3072 * 8 + 768 * 4)
        writes = [worker * 128 + logical for worker in range(6) for logical in range(128)]
        self.assertEqual(sorted(writes), list(range(768)))
        self.assertEqual((6144 // 32 + 191) // 192, 1)

    def test_source_and_launch_gate_exact_long_shape_and_preserve_thresholds(self):
        original = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8-double-buffer/build-v1/matmul_nbits.cc")
        selected = transform(original.read_text())
        self.assertIn("N_ == 2560 && K_ == 6144 &&", selected)
        self.assertIn(".SetWorkgroupSize(768)", selected)
        self.assertIn("maxComputeWorkgroupStorageSize >= 27648", selected)
        launch = controller_source((ROOT / "run_dense_tile_parallel2_long_k_v1.py").read_text())
        self.assertIn("threads = 768 if selected else 128", launch)
        self.assertIn("precise_duration_us(event)", launch)
        self.assertIn("ratio > 1.01 and positive >= 120", launch)
        self.assertIn("len(kernels) != 144", launch)
        self.assertIn('"selected": (k, n) == (6144, 2560)', launch)
        build = builder_source((ROOT / "build_dense_tile_parallel2_long_k_v1.py").read_text())
        self.assertIn('sha(library) != precise["provider_sha256"]', build)
        self.assertIn('"independent_tile_workers": 6', build)
        compile(build, "six-worker-builder", "exec")
        compile(launch, "six-worker-controller", "exec")


if __name__ == "__main__":
    unittest.main()
