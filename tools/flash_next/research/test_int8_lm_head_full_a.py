from pathlib import Path
import unittest

from build_int8_lm_head_full_a import CLASS, FLAG, PROGRAM, STORAGE_BYTES, transform
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


class Int8FullATests(unittest.TestCase):
    def test_admission_and_original_paths(self):
        source = SOURCE.read_text()
        candidate = transform(source)
        self.assertIn(f'static const int lm_head_full_a_selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}"',
                      candidate)
        self.assertIn("bits_ == 8 && K_ == 2560", candidate)
        self.assertIn("N_ == 248320", candidate)
        self.assertIn("K_ % block_size_ == 0", candidate)
        self.assertIn("a->Shape().Size() == K_", candidate)
        self.assertIn("!has_zero_points && bias == nullptr", candidate)
        self.assertIn("!context.EnableMatmulFp32Accumulation()", candidate)
        self.assertIn("compute, 0, lm_head_full_a ? 8", candidate)
        self.assertEqual(STORAGE_BYTES, 5632)
        compute_body = source[source.index("    if (dense_bitcast_route && block32_mode == 1)"):
                              source.index("  const bool exact_route")]
        self.assertIn(compute_body, candidate)
        self.assertEqual(source[source.index("Status ApplyMatMulNBits("):],
                         candidate[candidate.index("Status ApplyMatMulNBits("):])

    def test_original_stream_word_scale_order(self):
        visited = []
        for tile in range(5):
            for lane in range(32):
                chunk = tile * 32 + lane
                for word in range(4):
                    visited.extend(range(chunk * 16 + word * 4, chunk * 16 + word * 4 + 4))
        self.assertEqual(sorted(visited), list(range(2560)))
        for block_size in (16, 32, 64, 80, 128, 256, 512, 1280, 2560):
            for chunk in range(160):
                scales = {(chunk * 16 + offset) // block_size for offset in range(16)}
                self.assertEqual(scales, {(chunk * 16) // block_size})
        for lane in range(32):
            self.assertEqual([tile * 32 + lane for tile in range(5)],
                             list(range(lane, 160, 32)))

    def test_actual_shader_renderer(self):
        shader = render(PROGRAM, 0, False, False, "", FLAG,
                        class_name=CLASS, constructor_args=())
        self.assertEqual(shader.count("workgroupBarrier();"), 2)
        self.assertIn("array<vec4<f16>, 640>", shader)
        self.assertIn("partials[(slot + column_group) * 32u + stream] += sum;", shader)
        self.assertIn("sum += dot(tile_a[chunk * 4u + word], weights);", shader)
        self.assertIn("value += partials[local_idx * 32u + lane];", shader)


if __name__ == "__main__":
    unittest.main()
