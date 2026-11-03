from pathlib import Path
import unittest

from build_dense_columns8_spatial_double_buffer import CLASS, FLAG, PROGRAM, SOURCE, transform
from build_dense_columns8_double_buffer import PROGRAM as ORIGINAL_PROGRAM
from test_hc_projection_inner_unroll import render


class DenseColumns8SpatialTests(unittest.TestCase):
    def test_renderer_preserves_workgroup_count_and_one_column_per_thread(self):
        shader = render(PROGRAM, 1, False, False, helper="", flag=FLAG,
                        class_name=CLASS, constructor_args=())
        self.assertIn("let col0 = workgroup_idx * 8u + column_group", shader)
        self.assertNotIn("packed1", shader)
        self.assertIn("sum0 += dot(x0, w0) + dot(x1, w1)", shader)
        self.assertIn("i < 256u; i += 256u", shader)
        self.assertIn("value += partials[local_idx * 32u + stream]", shader)
        self.assertEqual(shader.count("workgroupBarrier();"), 2)

    def test_unique_activation_and_partial_writers(self):
        writers = [i for local in range(256) for i in range(local, 256, 256)]
        self.assertEqual(sorted(writers), list(range(256)))
        self.assertEqual(sorted((local // 32) * 32 + local % 32 for local in range(256)), list(range(256)))

    def test_original_parent_and_exact_admission(self):
        source = Path(str(SOURCE).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        candidate = transform(source)
        self.assertIn(ORIGINAL_PROGRAM, candidate)
        for text in (f"{FLAG} must be 0 or 1", "maxComputeWorkgroupSizeX >= 256",
                     "maxComputeInvocationsPerWorkgroup >= 256",
                     "maxComputeWorkgroupStorageSize >= 4608",
                     ".SetWorkgroupSize(256).SetDispatchGroupSize(N_ / 8)",
                     "columns8 ? (use_spatial8 ? 8 : 7)"):
            self.assertIn(text, candidate)
        self.assertTrue(candidate.endswith(source[source.index("\nStatus ApplyMatMulNBits("):]))

    def test_generated_gpu4_guards_and_uuid(self):
        from prepare_dense_columns8_spatial_double_buffer_proofs import batch_source
        for actual in (False, True):
            source = batch_source(actual)
            self.assertIn('"--gpu-index", "4"', source)
            self.assertIn("GPU-68170cb3-c9ea-a588-7558-ae82635d9173", source)
            self.assertNotIn("run_guard, UUID", source)
            self.assertIn(f'FLAG = "{FLAG}"', source)
            self.assertIn('"DenseColumns8SpatialDoubleBuffer"', source)


if __name__ == "__main__":
    unittest.main()
