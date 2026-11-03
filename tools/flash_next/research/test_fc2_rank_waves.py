from pathlib import Path
import unittest

from build_fc2_rank_waves import transform


class RankWaveTests(unittest.TestCase):
    def test_preserves_arithmetic_and_final_reduction(self):
        original = Path("/home/sayanshaw/repos/qwen38-int4-fc2-private-a/build-v1/qmoe_fc2_final_mix.inc").read_text()
        candidate = transform(original)
        for fragment in ("sum += dot(cached_a[i * 2u], w0) + dot(cached_a[i * 2u + 1u], w1)",
                         "partials[rank * 2u + c][lane] += sum",
                         "for (var i = 0u; i < 32u; i++) { value += partials[local_idx][i]; }",
                         "acc += router_value * partials[r * 2u + local_idx][0]"):
            self.assertIn(fragment, candidate)
        self.assertEqual(original.count("workgroupBarrier"), candidate.count("workgroupBarrier"))
        self.assertIn("let rank = local_idx / 32u + wave * 4u", candidate)
        self.assertIn("if (rank < uniforms.k)", candidate)
        self.assertIn("requires ORT_WEBGPU_QMOE_FC2_PRIVATE_A=1", candidate)


if __name__ == "__main__":
    unittest.main()
