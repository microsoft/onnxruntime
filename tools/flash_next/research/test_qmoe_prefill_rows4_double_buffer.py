from pathlib import Path
import unittest

from test_hc_projection_inner_unroll import render
from build_qmoe_prefill_rows4_double_buffer import (
    CLASS, FLAG, NAME, STORAGE_BYTES, make_program, transform, transform_replay,
)
from test_dense_columns8_double_buffer import Columns8DoubleBufferTests


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")
QMOE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc")


class PrefillDoubleBufferTests(unittest.TestCase):
    def test_source_and_admission(self):
        source = SOURCE.read_text()
        program = make_program(source)
        transformed = transform(source)
        self.assertIn(program, transformed)
        self.assertIn(source[source.index("  if (grouped_prefill_rows == 1"):
                             source.index("\n  if (nbits == 3) {",
                                          source.index("  if (grouped_prefill_rows == 1"))], transformed)
        self.assertIn("prefill_double_buffer == 1 && K == 2560 && N == 1280", transformed)
        self.assertIn(f">= {STORAGE_BYTES}", transformed)
        self.assertEqual(STORAGE_BYTES, 18448)
        self.assertIn(f'static const int prefill_double_buffer = ParseEnvironmentVariableWithDefault<int>("{FLAG}"',
                      transformed)
        self.assertIn("SetWorkgroupSize(128).SetDispatchGroupSize(N / 8, M / 4)", transformed)
        replay = transform_replay(QMOE.read_text())
        self.assertEqual(replay.count(f'"{FLAG} must be 0 or 1"'), 2)
        self.assertIn("key.push_back(prefill_double_buffer);", replay)

    def test_shader_arithmetic_and_storage(self):
        source = SOURCE.read_text()
        start = source.index("class GroupedPrefillRows4Program final")
        stop = source.index("class MatMulInt3Program final", start)
        for bias in (False, True):
            original = render(source[start:stop], 0, False, False, "", FLAG,
                              class_name="GroupedPrefillRows4Program", constructor_args=(bias,))
            candidate = render(make_program(source), 0, False, False, "", FLAG,
                               class_name=CLASS, constructor_args=(bias,))
            self.assertIn("array<array<vec4<f16>, 1024>, 2>", candidate)
            candidate = candidate.replace("array<array<vec4<f16>, 1024>, 2>",
                                          "array<vec4<f16>, 1024>")
            candidate = candidate.replace("  let buffer = (kidx / 1024u) % 2u;\n", "")
            candidate = candidate.replace("tile_a[buffer]", "tile_a")
            candidate = candidate.replace("}\nworkgroupBarrier();\nif (local_idx < 8u)",
                                          "  workgroupBarrier();\n}\nif (local_idx < 8u)")
            self.assertEqual(original, candidate)
            self.assertNotIn(NAME, original)

    def test_buffer_protocol(self):
        Columns8DoubleBufferTests.test_randomized_warp_scheduling_preserves_buffers(self)


if __name__ == "__main__":
    unittest.main()
