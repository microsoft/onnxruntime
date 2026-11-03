from pathlib import Path
import unittest

from build_hc_up_tiled import tiled_program, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-up-premix/build-v2/hc_up_premix.cc")


class HCUpTiledTests(unittest.TestCase):
    def test_complete_column_and_branch_mapping(self):
        source = SOURCE.read_text()
        result = tiled_program(source)
        self.assertIn("partials: array<f16, 1024>", result)
        self.assertIn("gates: array<f16, 32>", result)
        self.assertIn("partials[offset * 128u + c * 32u + i]", result)
        self.assertIn("gates[local_idx * 4u + c]", result)
        self.assertIn('y.SetByOffset("base_col + local_idx",', result)
        for line in source.splitlines():
            if any(expression in line for expression in ("sum += dot", "total += sum", "let w0", "let w1")):
                self.assertIn(line, result)
        outputs = [group * 8 + lane for group in range(320) for lane in range(8)]
        self.assertEqual(outputs, list(range(2560)))
        partials = [offset * 128 + branch * 32 + lane
                    for offset in range(8) for branch in range(4) for lane in range(32)]
        self.assertEqual(partials, list(range(1024)))

    def test_original_fallback_and_replay_variant_preserved(self):
        source = SOURCE.read_text()
        result = transform(source)
        self.assertIn(source[source.index("class HCUpPreMixProgram"):source.index("class HCUpPreMixF16")], result)
        self.assertIn(source[source.index("    auto projected"):], result)
        self.assertIn("ORT_WEBGPU_HC_UP_TILED must be 0 or 1", result)
        self.assertIn(".SetDispatchGroupSize(tiled == 1 ? 320 : 2560)", result)
        self.assertIn("}, 0, tiled);", result)


if __name__ == "__main__":
    unittest.main()
