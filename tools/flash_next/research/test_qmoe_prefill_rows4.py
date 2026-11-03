from pathlib import Path
import unittest

from build_qmoe_prefill_rows4 import FLAG, transform
from build_qmoe_prefill_rows4 import PROGRAM
from run_qsa_prefill_score16_model import rows4_program_count


class PrefillRows4Tests(unittest.TestCase):
    def test_private_route_leaves_original_apply_and_compute_bodies_intact(self):
        original = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/matmul_nbits.cc").read_text()
        actual = transform(original)
        start = original.index("Status ApplyMatMulNBits(")
        body = original.index("  if (nbits == 3) {", start)
        self.assertIn(original[body:], actual)
        self.assertEqual(actual.count("GroupedPrefillRows4Program program"), 1)
        self.assertIn("(M == 5120 || M == 10240)", actual)
        self.assertIn("(K == 2560 && N == 1280) || (K == 640 && N == 2560)", actual)
        self.assertIn("{a, ProgramTensorMetadataDependency::TypeAndRank, 4}", actual)
        self.assertIn("{b, ProgramTensorMetadataDependency::TypeAndRank, 16}", actual)
        self.assertIn("has_weight_idx_indirect && !has_zero_points", actual)
        self.assertIn("!context.EnableMatmulFp32Accumulation()", actual)
        self.assertIn(f"{FLAG} must be 0 or 1", actual)

    def test_partial_ownership_matches_original_32_lane_k_partition(self):
        for dimension in (640, 2560):
            for k in range(dimension):
                tile, within = divmod(k, 1024)
                lane, product = divmod(within, 32)
                self.assertEqual(tile * 1024 + lane * 32 + product, k)
        for rows in (5120, 10240):
            emitted = [tile * 4 + row for tile in range(rows // 4) for row in range(4)]
            self.assertEqual(emitted, list(range(rows)))
        program = Path(__file__).with_name("qmoe_prefill_rows4.inc").read_text()
        self.assertIn("partials[row][slot + idy][idx] += sum;", program)
        self.assertIn("value += partials[row][local_idx][lane];", program)
        self.assertIn("expert != cached_expert", program)
        self.assertIn("kidx += 1024u", program)
        self.assertIn("word < 4u; word++", program)
        self.assertEqual(program.count("workgroupBarrier();"), 2)
        self.assertNotIn("subgroupMatrix", program)

    def test_actual_profile_requires_both_correct_vectorized_projections(self):
        events = [{"cat": "Kernel", "name": "QMoE&" + PROGRAM, "args": {"shapes":
                   f"inputs[0] = {{10240,{width}}} outputs[0] = {{10240,{columns}}} "}}
                  for width, columns in ((640, 1280), (160, 2560))]
        self.assertEqual(rows4_program_count(events, 1), 2)
        for invalid in ([], events[:1], events + events[:1], events[:1] * 2):
            with self.assertRaises(AssertionError):
                rows4_program_count(invalid, 1)
        invalid_shape = [{"cat": "Kernel", "name": "QMoE&" + PROGRAM,
                          "args": {"shapes": "inputs[0] = {10240,640} outputs[0] = {10240,2560}"}}]
        with self.assertRaises(AssertionError):
            rows4_program_count(invalid_shape, 1)


if __name__ == "__main__":
    unittest.main()
