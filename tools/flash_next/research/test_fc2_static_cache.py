from pathlib import Path
import unittest

from build_fc2_static_cache import transform


class StaticCacheTests(unittest.TestCase):
    def test_keeps_word_addition_and_original_fallback(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-fc2-private-a/build-v1/qmoe_fc2_final_mix.inc").read_text()
        candidate = transform(source)
        self.assertIn('const char* components[] = {"x", "y", "z", "w"}', candidate)
        self.assertIn("for (int i = 0; i < 4; ++i)", candidate)
        self.assertIn("sum += dot(cached_a", candidate)
        self.assertIn("var cached_a: array<vec4<f16>, 8>", candidate)
        self.assertIn("partials[rank * 2u + c][lane] += sum", candidate)
        self.assertIn("acc += router_value * partials[r * 2u + local_idx][0]", candidate)
        self.assertEqual(source.count("workgroupBarrier"), candidate.count("workgroupBarrier"))
        self.assertIn("requires ORT_WEBGPU_QMOE_FC2_PRIVATE_A=1", candidate)


if __name__ == "__main__":
    unittest.main()
