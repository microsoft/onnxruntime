import ast
from pathlib import Path
import unittest

from hc_gate_staging_transform import CLASS, FLAG, make_program, transform
from prepare_hc_gate_staging import ROOT, build_source, controller_source
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


class GateStageTests(unittest.TestCase):
    def test_actual_renderer_changes_only_gate_prologue(self):
        from build_hc_projection_inner_unroll import program_parts
        original = program_parts(SOURCE.read_text())[2]
        changed = make_program(SOURCE.read_text())
        for shared, scale in ((False, False), (True, False), (True, True)):
            before = render(original, 0, shared, scale)
            after = render(changed, 1, shared, scale, flag=FLAG, class_name=CLASS)
            marker = "for (var h = local_idx;"
            self.assertEqual(before[before.index(marker):], after[after.index(marker):])
            self.assertIn("if (local_idx < 4u)", after)
            self.assertIn("staged_gates[local_idx] = f32(gate_value)", after)
            self.assertEqual("if (local_idx == 4u)" in after, shared)
            self.assertEqual(after.count("workgroupBarrier();"), before.count("workgroupBarrier();") + 1)

    def test_original_default_fallbacks_geometry_and_static_replay_preserved(self):
        original = SOURCE.read_text()
        changed = transform(original)
        start = original.index("class HCPostNormProjectionProgram final")
        stop = original.index("class HCPostNormProjectionF16", start)
        self.assertIn(original[start:stop], changed)
        suffix = original.index("    HCPostNormProjectionBaselineNormProgram")
        self.assertEqual(changed[changed.index("    HCPostNormProjectionBaselineNormProgram"):], original[suffix:])
        self.assertIn(f'static const int selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)', changed)
        self.assertIn("selector == 0 || selector == 1", changed)
        self.assertIn(".SetWorkgroupSize(256).SetDispatchGroupSize(41)", changed)
        self.assertNotIn("}, 0, gate_staging)", changed)

    def test_generated_build_and_controller_are_fresh_and_correctly_scoped(self):
        builder = build_source((ROOT / "build_hc_owned_pair34.py").read_text())
        controller = controller_source((ROOT / "run_hc_owned_pair34_v1.py").read_text())
        ast.parse(builder)
        ast.parse(controller)
        self.assertIn("from hc_gate_staging_transform import transform", builder)
        self.assertIn('"shared_storage_bytes": 27668', builder)
        self.assertIn('"gates_computed_once_per_workgroup": True', builder)
        self.assertNotIn('"single_owner_pair_stores": True', builder)
        self.assertIn('VARIANT = "hc-gate-staging-v1"', controller)
        self.assertIn('NAME = "HCPostNormProjectionGateStagingF16"', controller)
        self.assertIn("qwen38-int4-fc2-private-a/build-v1", builder)
        self.assertIn('"original_hc_object_byte_identical": True', builder)


if __name__ == "__main__":
    unittest.main()
