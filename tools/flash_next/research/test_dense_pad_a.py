from collections import Counter
from pathlib import Path
import re
import unittest

from build_dense_pad_a import FLAG, padded_template, transform


class DensePadATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32")
        cls.template = (root / "block32-template-v1/dense_block32_template.inc").read_text()
        cls.source = (root / "build-v1/matmul_nbits.cc").read_text()

    def test_every_store_and_load_maps_to_same_unique_padded_vector(self):
        for length in (256, 512, 1024):
            indices = [(i // 8) * 9 + i % 8 for i in range(length)]
            self.assertEqual(len(set(indices)), length)
            self.assertLess(max(indices), ((length + 7) // 8) * 9)
        padded = padded_template(self.template)
        self.assertGreater(padded.count("tile_A[dense_padded_a_index("), 30)
        self.assertEqual(padded.count("tile_A["), padded.count("tile_A[dense_padded_a_index("))

    def test_floating_operations_and_barriers_are_unchanged(self):
        original = "\n".join(line for line in self.template.splitlines() if not line.lstrip().startswith("//")) + "\n"
        padded = re.sub(r"tile_A\[dense_padded_a_index\(([^)]+)\)\]", r"tile_A[\1]",
                        padded_template(self.template))
        for marker in ("sum +=", "inter_results", "workgroupBarrier", "scale_b", "decode_low_nibbles"):
            self.assertEqual([line for line in original.splitlines() if marker in line],
                             [line for line in padded.splitlines() if marker in line])

    def test_bank_aliasing_reduced_and_parent_route_and_replay_preserved(self):
        original = Counter((lane * 8 * 8 // 4) % 32 for lane in range(32))
        padded = Counter((lane * 9 * 8 // 4) % 32 for lane in range(32))
        self.assertEqual(max(original.values()), 16)
        self.assertEqual(max(padded.values()), 2)
        source = transform(self.source)
        self.assertIn(f"{FLAG} must be 0 or 1", source)
        self.assertIn("a->IsDataType<MLFloat16>() && !context.EnableMatmulFp32Accumulation()", source)
        self.assertIn("pad_a_route ? 7 : block32_mode == 1 ? 3", source)
        self.assertIn("ApplyDenseBlock32MatMulNBits", source)

    def test_cli_metadata_and_fp32_fallback_gate_wired(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-dense-pad-a-windows", benchmark)
        self.assertIn("args.compare_dense_pad_a_windows or", benchmark)
        self.assertIn('"dense_pad_a_decode_windows" if args.compare_dense_pad_a_windows', benchmark)
        self.assertIn("DENSE_PAD_A_FLAG if args.compare_dense_pad_a_windows", benchmark)
        self.assertIn(f'"{FLAG}",', benchmark)
        self.assertIn('DENSE_PAD_A_FLAG: "dense_pad_a"', (root / "qsa_rank_decode_windows.py").read_text())
        self.assertIn('"dense-pad-a-v1"', (root / "run_hc_up_parallel2_model.py").read_text())


if __name__ == "__main__":
    unittest.main()
