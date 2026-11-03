from pathlib import Path
import unittest

from build_hc_up_parallel2 import FLAG, packed_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-up-premix/build-v2/hc_up_premix.cc")


class HCUpParallel2Tests(unittest.TestCase):
    def test_partial_writes_are_complete_disjoint_and_keep_original_slots(self):
        program = packed_program(SOURCE.read_text())
        for expression in ("let lane = local_idx % 16u;",
                           "let branch = (local_idx / 16u) % 4u;",
                           "let column = local_idx / 64u;",
                           "let partial_idx = column * 128u + branch * 32u + lane;",
                           "partials[partial_idx] = total;",
                           "partials[partial_idx + 16u] = f16(0);"):
            self.assertIn(expression, program)
        writes = []
        gates = []
        for thread in range(128):
            column, branch, lane = thread // 64, (thread // 16) % 4, thread % 16
            start = column * 128 + branch * 32
            writes.extend((start + lane, start + lane + 16))
            if lane == 0:
                gates.append(column * 4 + branch)
            for slot in (lane, lane + 16):
                original_value = ("block", slot) if slot < 10 else 0
                packed_value = ("block", lane) if slot == lane and lane < 10 else 0
                self.assertEqual(packed_value, original_value)
        self.assertEqual(sorted(writes), list(range(256)))
        self.assertEqual(gates, list(range(8)))

    def test_every_output_and_projection_column_has_one_writer(self):
        program = packed_program(SOURCE.read_text())
        for expression in ("let output_col = workgroup_idx * 2u + column;",
                           "let col = output_col + branch * 2560u;",
                           "if (branch == 0u && lane == 0u)",
                           'y.SetByOffset("output_col",'):
            self.assertIn(expression, program)
        outputs, projections = [], []
        for group in range(1280):
            for thread in range(128):
                column, branch, lane = thread // 64, (thread // 16) % 4, thread % 16
                output = group * 2 + column
                if branch == 0 and lane == 0:
                    outputs.append(output)
                if lane == 0:
                    projections.append(output + branch * 2560)
        self.assertEqual(outputs, list(range(2560)))
        self.assertEqual(sorted(projections), list(range(10240)))

    def test_shader_preserves_arithmetic_and_uniform_barriers(self):
        source = SOURCE.read_text()
        program = packed_program(source)
        for expression in (
                "if (lane < 10u)", "let packed", "let scale", "let low", "let high",
                "let w0", "let w1", "let x0", "let x1", "sum += dot", "total += sum",
                "for (var i = 0u; i < 32u;", "gates[branch] = f16",
                "for (var c = 0u; c < uniforms.branches;", "sum * uniforms.reduction_scale"):
            for line in source[:source.index("class HCUpPreMixF16")].splitlines():
                if expression in line:
                    expected = line.replace("partials[local_idx + i]", "partials[partial_idx + i]")
                    expected = expected.replace("gates[branch]", "gates[column * 4u + branch]")
                    expected = expected.replace('y.SetByOffset("workgroup_idx",',
                                                'y.SetByOffset("output_col",')
                    self.assertIn(expected, program)
        self.assertEqual(program.count("workgroupBarrier()"), 2)
        self.assertNotIn("return;", program)
        self.assertIn("partials[partial_idx + 16u] = f16(0)", program)
        self.assertIn("partials: array<f16, 256>", program)
        self.assertIn("gates: array<f16, 8>", program)
        self.assertIn('Program{"HCUpPreMixPacked2F16"}', program)
        self.assertNotIn("for (var offset", program)

    def test_original_shader_fallback_and_replay_layout_are_preserved(self):
        source = SOURCE.read_text()
        actual = transform(source)
        start, stop = source.index("class HCUpPreMixProgram"), source.index("class HCUpPreMixF16")
        self.assertIn(source[start:stop], actual)
        self.assertIn(source[source.index("    auto projected"):], actual)
        self.assertIn(source[source.index("\n private:"):], actual)
        self.assertIn(f'"{FLAG}", 0', actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertLess(actual.index(f'"{FLAG}"'), actual.index('const int enabled'))
        self.assertIn(".SetDispatchGroupSize(parallel2 == 1 ? 1280 : 2560)", actual)
        self.assertIn("}, 0, parallel2);", actual)
        self.assertIn("params.rows == 1 && !context.EnableMatmulFp32Accumulation()", actual)


if __name__ == "__main__":
    unittest.main()
