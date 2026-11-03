from pathlib import Path
import unittest

from dense_tile_parallel3_transform import PROGRAM as THREE_PROGRAM
from dense_tile_parallel4_long_k_transform import NAME, PROGRAM, SHARED_BYTES, transform
from prepare_dense_tile_parallel4_long_k_ns_v1 import builder_source, controller_source


ROOT = Path(__file__).parent


class FourWorkerTests(unittest.TestCase):
    def test_original_arithmetic_and_tail_write_ownership(self):
        normalized = PROGRAM.replace(NAME, "DenseTileParallel3Packed")
        for before, after in (
            ("tile_a: array<vec4<f16>, 2048>", "tile_a: array<vec4<f16>, 1536>"),
            ("wave * 128u", "wave * 96u"),
            ("wave * 4u + tile_worker", "wave * 3u + tile_worker"),
            ("(wave % 2u) * 4u + tile_worker", "(wave % 2u) * 3u + tile_worker"),
            ("  if (tile * 32u < uniforms.blocks) {\n"
             "    tile_sums[tile * 128u + logical_idx] = bitcast<u32>(vec2<f16>(sum0, sum1));\n"
             "  }",
             "  tile_sums[tile * 128u + logical_idx] = bitcast<u32>(vec2<f16>(sum0, sum1));"),
        ):
            normalized = normalized.replace(before, after)
        self.assertEqual(normalized, THREE_PROGRAM)
        writes = [tile * 128 + logical for wave in range(2) for worker in range(4)
                  if (tile := wave * 4 + worker) < 6 for logical in range(128)]
        self.assertEqual(sorted(writes), list(range(768)))
        self.assertEqual(SHARED_BYTES, 2048 * 8 + 768 * 4)

    def test_source_and_native_timing_gate(self):
        original = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8-double-buffer/build-v1/matmul_nbits.cc")
        selected = transform(original.read_text())
        self.assertIn("N_ == 2560 && K_ == 6144 &&", selected)
        self.assertIn(".SetWorkgroupSize(512)", selected)
        self.assertIn("maxComputeWorkgroupStorageSize >= 19456", selected)
        build = builder_source((ROOT / "build_dense_tile_parallel6_long_k_ns_v1.py").read_text())
        launch = controller_source((ROOT / "run_dense_tile_parallel6_long_k_ns_v1.py").read_text())
        self.assertIn('"physical_threads": 512, "independent_tile_workers": 4', build)
        self.assertIn('sha(library) != precise["provider_sha256"]', build)
        self.assertIn("threads = 512 if selected else 128", launch)
        self.assertIn("ratio > 1.01 and positive >= 120", launch)
        self.assertIn("precise_duration_us(event)", launch)
        compile(build, "four-worker-builder", "exec")
        compile(launch, "four-worker-controller", "exec")


if __name__ == "__main__":
    unittest.main()
