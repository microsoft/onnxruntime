from pathlib import Path
import unittest

from build_hc_separate8 import FLAG, NORM_CLASS, PROJECTION_CLASS, SOURCE, programs, transform
from test_hc_projection_inner_unroll import render


class HCSeparate8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(str(SOURCE).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        cls.norm, cls.projection = programs(cls.source)
        cls.candidate = transform(cls.source)

    def test_original_classes_and_fallback_preserved(self):
        start = self.source.index("class HCPostNormProjectionBaselineNormProgram")
        end = self.source.index("class HCPostNormProjectionF16", start)
        self.assertIn(self.source[start:end], self.candidate)
        fallback = self.source[self.source.index("    HCPostNormProjectionBaselineNormProgram norm{"):]
        self.assertTrue(self.candidate.endswith(fallback))
        self.assertIn(f'static const int separate8 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)',
                      self.candidate)

    def test_exact_norm_cpp_renderer(self):
        start = self.source.index("class HCPostNormProjectionBaselineNormProgram")
        end = self.source.index("class HCPostNormProjectionBaselineMatMulProgram", start)
        original = self.source[start:end]
        for shared in (False, True):
            for shared_scale in (False, True):
                arguments = (shared, shared_scale, 2560)
                first = render(original, 0, shared, shared_scale, helper="",
                               class_name="HCPostNormProjectionBaselineNormProgram", constructor_args=arguments)
                second = render(self.norm, 1, shared, shared_scale, helper="", flag=FLAG,
                                class_name=NORM_CLASS, constructor_args=arguments)
                self.assertEqual(first, second)

    def test_projection_renderer_and_mapping(self):
        shader = render(self.projection, 1, False, False, helper="", flag=FLAG,
                        class_name=PROJECTION_CLASS, constructor_args=())
        self.assertIn("partials: array<f16, 256>", shader)
        self.assertIn("if (col < 324u)", shader)
        self.assertIn("if (lane == 0u && col < 324u)", shader)
        self.assertIn("sum += dot(x0, w0) + dot(x1, w1)", shader)
        self.assertIn("value += partials[local_idx + i]", shader)
        self.assertEqual(shader.count("{"), shader.count("}"))
        columns = [group * 8 + local // 32 for group in range(41)
                   for local in range(256) if local % 32 == 0 and group * 8 + local // 32 < 324]
        self.assertEqual(columns, list(range(324)))

    def test_combined_replay_and_no_kernel_layout_changes(self):
        self.assertEqual(self.candidate.count("return replay_.Run("), self.source.count("return replay_.Run("))
        self.assertEqual(self.candidate.count("mutable StaticNodeReplay replay_;"), 1)
        self.assertNotIn("HCStageReplay", self.candidate)
        self.assertNotIn("CreateGPUTensor", self.candidate[:self.candidate.index("    auto intermediate =")])
        stage = self.candidate[self.candidate.index("        if (separate8 == 1)"):]
        self.assertLess(stage.index("context.RunProgram(norm)"), stage.index("context.RunProgram(program)"))
        self.assertIn(".SetWorkgroupSize(256).SetDispatchGroupSize(41)", stage)
        self.assertIn("{normalized, ProgramTensorMetadataDependency::TypeAndRank, 4}", stage)


if __name__ == "__main__":
    unittest.main()
