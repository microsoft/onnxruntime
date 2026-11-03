from pathlib import Path
import unittest

from build_dense_columns16_spatial_double_buffer import CLASS, FLAG, PROGRAM, SOURCE, transform
from build_dense_columns8_double_buffer import PROGRAM as ORIGINAL_PROGRAM
from test_hc_projection_inner_unroll import render


class DenseColumns16SpatialTests(unittest.TestCase):
    def test_actual_renderer_one_column_per_thread_and_original_rounding(self):
        shader = render(PROGRAM, 1, False, False, helper="", flag=FLAG,
                        class_name=CLASS, constructor_args=())
        self.assertIn("let col0 = workgroup_idx * 16u + column_group", shader)
        self.assertNotIn("packed1", shader)
        self.assertNotIn("sum1", shader)
        self.assertIn("sum0 += dot(x0, w0) + dot(x1, w1)", shader)
        self.assertIn("partials[(column_group + 0u) * 32u + stream] += sum0", shader)
        self.assertIn("value += partials[local_idx * 32u + stream]", shader)
        self.assertEqual(shader.count("workgroupBarrier();"), 2)
        self.assertIn("i < 256u; i += 512u", shader)

    def test_unique_partial_and_activation_writers(self):
        partials, activations = [], []
        for local in range(512):
            partials.append((local // 32) * 32 + local % 32)
            activations.extend(range(local, 256, 512))
        self.assertEqual(sorted(partials), list(range(512)))
        self.assertEqual(sorted(activations), list(range(256)))

    def test_original_parent_and_exact_hardware_admission(self):
        source = Path(str(SOURCE).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        candidate = transform(source)
        self.assertIn(ORIGINAL_PROGRAM, candidate)
        for text in (f"{FLAG} must be 0 or 1", "N_ % 16 == 0",
                     "maxComputeWorkgroupSizeX >= 512",
                     "maxComputeInvocationsPerWorkgroup >= 512",
                     "maxComputeWorkgroupStorageSize >= 5120",
                     ".SetWorkgroupSize(512).SetDispatchGroupSize(N_ / 16)",
                     "columns8 ? (use_spatial16 ? 8 : 7)"):
            self.assertIn(text, candidate)
        self.assertTrue(candidate.endswith(source[source.index("\nStatus ApplyMatMulNBits("):]))

    def test_unprofiled_comparator_retains_exact_checks_and_warmup_exclusion(self):
        from build_dense_columns16_spatial_double_buffer import ROOT, timing_comparator
        source = timing_comparator((ROOT / "dense_paired_matrix_compare.cpp").read_text())
        self.assertIn("if (!timing) options.EnableProfiling(argv[5])", source)
        self.assertIn("iteration = -8", source)
        self.assertIn("if (iteration >= 0)", source)
        self.assertIn("consumed != std::strlen(argv[5])", source)
        self.assertIn("std::memcmp(&reference[i], &observed[i]", source)
        self.assertIn("profiling=0", source)

    def test_generated_controllers_and_matched_timing_direction(self):
        from prepare_dense_columns16_spatial_double_buffer_proofs import batch_source
        from run_dense_columns16_spatial_double_buffer_unprofiled import matched_shape
        for actual in (False, True):
            source = batch_source(actual)
            self.assertIn(f'FLAG = "{FLAG}"', source)
            self.assertIn('FLAGS += " ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER=1"', source)
            self.assertIn('"DenseColumns16SpatialDoubleBuffer"', source)
        rows = [dict(before=dict(mean_us=100), candidate=dict(mean_us=80),
                     after=dict(mean_us=100))] * 3
        result = matched_shape(rows)
        self.assertEqual(result["positive"], 3)
        self.assertEqual(result["matched_speed_ratio"], 1.25)


if __name__ == "__main__":
    unittest.main()
