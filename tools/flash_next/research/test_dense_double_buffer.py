from pathlib import Path
import random
import unittest

from build_dense_double_buffer import STORAGE_BYTES, double_buffer_template, private_program, transform


BASE = Path("/home/sayanshaw/repos")
SOURCE = BASE / "qwen38-int4-dense-tile4/build-v1/matmul_nbits.cc"
TEMPLATE = BASE / "qwen38-int4-dense-bitcast-unpack/build-v1/dense_bitcast_template.inc"


class DoubleBufferTests(unittest.TestCase):
    def test_only_activation_buffer_addresses_and_barrier_placement_change(self):
        source = TEMPLATE.read_text()
        actual = double_buffer_template(source)
        self.assertIn('array<array<input_a_value_t, ', actual)
        self.assertIn('ss << ">, 2>;\\n";', actual)
        self.assertEqual(actual.count('tile_A[(kidx / " << __param_tile_size_k << "u) & 1u]['),
                         sum(line.count("tile_A[") for line in source.splitlines()
                             if not line.lstrip().startswith("//")))
        for line in source.splitlines():
            if line.startswith('ss << "') and any(token in line for token in
                    ("sum +=", "inter_results[", "output_value +=", "decode_low_nibbles")):
                self.assertIn(line.replace("tile_A[", 'tile_A[(kidx / " << __param_tile_size_k << "u) & 1u]['),
                              actual)
        self.assertIn('ss << "  }\\n";\nss << "  workgroupBarrier();\\n";', actual)
        program = private_program(SOURCE.read_text())
        self.assertIn("a_length_per_tile = tile_size_k / components_a", program)
        self.assertEqual(STORAGE_BYTES, 4352)

    def test_previous_iteration_barrier_protects_reused_buffer(self):
        for tiles in (1, 3, 6, 10):
            for seed in range(16):
                rng = random.Random(seed)
                iteration = [0] * 4
                phase = ["load"] * 4
                completed = [-1] * 4
                steps = 0
                while any(stage != "done" for stage in phase):
                    ready = [warp for warp in range(4) if phase[warp] in ("load", "compute")]
                    self.assertTrue(ready)
                    warp = rng.choice(ready)
                    if phase[warp] == "load":
                        self.assertTrue(all(done >= iteration[warp] - 2 for done in completed))
                        phase[warp] = "barrier"
                        if all(stage == "barrier" for stage in phase):
                            self.assertEqual(len(set(iteration)), 1)
                            phase = ["compute"] * 4
                    else:
                        completed[warp] = iteration[warp]
                        iteration[warp] += 1
                        phase[warp] = "load" if iteration[warp] < tiles else "done"
                    steps += 1
                    self.assertLessEqual(steps, 8 * tiles)
                self.assertEqual(completed, [tiles - 1] * 4)

    def test_original_program_and_fallbacks_remain_with_distinct_replay_key(self):
        source = SOURCE.read_text()
        actual = transform(source)
        start = source.index("class DenseTile4Program")
        stop = source.index("\n}\n\nStatus MatMulNBitsProgram::", start)
        self.assertIn(source[start:stop], actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= double_buffer_storage_bytes", actual)
        self.assertIn("context.EnableMatmulFp32Accumulation() ? sizeof(float) : sizeof(MLFloat16)", actual)
        self.assertIn("double_buffer ? 3 : 1 + dense_tile4", actual)
        self.assertIn("dense_bitcast_route && double_buffer", actual)


if __name__ == "__main__":
    unittest.main()
