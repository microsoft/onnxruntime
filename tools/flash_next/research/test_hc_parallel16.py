import unittest
from pathlib import Path

from build_hc_parallel16 import CLASS, FLAG, STORAGE_BYTES, make_program, transform
from build_hc_projection_inner_unroll import SOURCE, program_parts
from test_hc_projection_inner_unroll import render


class HCParallel16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(str(SOURCE).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        cls.program = make_program(cls.source)
        cls.candidate = transform(cls.source)

    def test_original_and_fallbacks_preserved(self):
        _, _, original = program_parts(self.source)
        self.assertIn(original, self.candidate)
        fallback = self.source[self.source.index("    HCPostNormProjectionBaselineNormProgram norm"):]
        self.assertTrue(self.candidate.endswith(fallback))
        self.assertIn("      });", self.candidate)
        self.assertNotIn("}, 0, parallel16)", self.candidate)

    def test_selector_and_device_admission(self):
        for text in (
            f'static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)',
            "selector == 0 || selector == 1",
            'context.AdapterInfo().vendor == std::string_view{"nvidia"}',
            "limits.maxComputeWorkgroupSizeX >= 512",
            "limits.maxComputeInvocationsPerWorkgroup >= 512",
            f"limits.maxComputeWorkgroupStorageSize >= {STORAGE_BYTES}",
            ".SetWorkgroupSize(parallel16 ? 512 : 256).SetDispatchGroupSize(parallel16 ? 21 : 41)",
        ):
            self.assertIn(text, self.candidate)
        self.assertEqual(STORAGE_BYTES, 28160)

    def test_actual_cpp_renderer_and_uniform_barriers(self):
        for shared in (False, True):
            for shared_scale in (False, True):
                with self.subTest(shared=shared, shared_scale=shared_scale):
                    shader = render(self.program, 1, shared, shared_scale, helper="",
                                    flag=FLAG, class_name=CLASS)
                    self.assertIn("partials: array<f16, 512>", shader)
                    self.assertIn("let col = workgroup_idx * 16u + local_idx / 32u;", shader)
                    self.assertEqual(shader.count("if (local_idx < 256u) {"), 3)
                    self.assertIn("for (var width = 128u; width > 0u; width >>= 1u)", shader)
                    start = 0
                    for _ in range(3):
                        start = shader.index("if (local_idx < 256u) {", start)
                        opening = shader.index("{", start)
                        depth = 1
                        end = opening + 1
                        while depth:
                            depth += (shader[end] == "{") - (shader[end] == "}")
                            end += 1
                        self.assertNotIn("workgroupBarrier", shader[opening:end])
                        start = end
                    self.assertEqual(shader.count("{"), shader.count("}"))
                    self.assertIn("value += partials[local_idx + i]", shader)

    def test_original_rms_and_staging_ownership(self):
        hidden = 2560
        original = [(stream * hidden + h, local)
                    for stream in range(4) for local in range(256)
                    for h in range(local, hidden, 256)]
        candidate = [(stream * hidden + h, local)
                     for stream in range(4) for local in range(512) if local < 256
                     for h in range(local, hidden, 256)]
        self.assertEqual(candidate, original)
        self.assertEqual(len({index for index, _ in candidate}), hidden * 4)
        sums = [stream * 256 + local for stream in range(4)
                for local in range(512) if local < 256]
        self.assertEqual(sorted(sums), list(range(1024)))

    def test_projection_lane_schedule_and_tail(self):
        columns = []
        for group in range(21):
            for local in range(512):
                col, lane = group * 16 + local // 32, local % 32
                if col >= 324:
                    continue
                original_local = (col % 8) * 32 + lane
                self.assertEqual([tile * 32 + lane for tile in range(10)],
                                 [tile * 32 + original_local % 32 for tile in range(10)])
                if lane == 0:
                    columns.append(col)
                    self.assertLessEqual(local + 31, 511)
        self.assertEqual(columns, list(range(324)))
        self.assertNotIn("for (var slot", self.program)
        self.assertIn('<< "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\\n"',
                      self.program)


if __name__ == "__main__":
    unittest.main()
