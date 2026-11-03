from pathlib import Path
import unittest

from build_qmoe_row4_static_weights import (
    FLAG, HELPER, comparator_source, program_parts, static_program, transform_program, transform_source,
)
from test_hc_projection_inner_unroll import render


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")


def expected_shader(original):
    original = original.replace("var weights: array<vec4<f16>, 8>;",
                                "\n".join(f"var weights{index}: vec4<f16>;" if index == 0 else
                                          f"      var weights{index}: vec4<f16>;" for index in range(8)))
    for indent, next_line in ((10, "          cached_expert"), (8, "        partials")):
        prefix = " " * indent
        start = original.index(prefix + "for (var word = 0u; word < 4u; word++)")
        body_start = original.index("\n", start) + 1
        end = original.index(next_line, body_start)
        closing = prefix + "}\n"
        assert original[start:end].endswith(closing)
        body = original[body_start:end - len(closing)]
        steps = []
        for word in range(4):
            step = body.replace("packed[word]", f"packed[{word}u]")
            step = step.replace("weights[word * 2u + 1u]", f"weights{word * 2 + 1}")
            step = step.replace("weights[word * 2u]", f"weights{word * 2}")
            step = step.replace("word * 2u", f"{word * 2}u")
            steps.append(prefix + "{\n" + step + closing)
        original = original[:start] + "".join(steps) + original[end:]
    return original


class QmoeRow4StaticWeightsTests(unittest.TestCase):
    def test_actual_emitter_default_equality_and_exact_static_steps(self):
        program = program_parts(SOURCE.read_text())[2]
        for bias in (False, True):
            arguments = dict(helper=HELPER, flag=FLAG, class_name="GroupedPrefillRows4Program",
                             constructor_args=(bias,))
            original = render(program, 0, False, False, **arguments)
            self.assertEqual(render(transform_program(program), 0, False, False, **arguments), original)
            candidate = render(transform_program(program), 1, False, False, **arguments)
            self.assertEqual(candidate, expected_shader(original))
            self.assertEqual(candidate.count("sum += dot("), 4)
            self.assertEqual(candidate.count("workgroupBarrier();"), original.count("workgroupBarrier();"))
            self.assertNotIn("weights[", candidate)
            self.assertNotIn("var word", candidate)

    def test_compute_admission_and_fallbacks_unchanged(self):
        source = SOURCE.read_text()
        start, end, _ = program_parts(source)
        candidate = transform_source(source)
        self.assertTrue(candidate.endswith(source[end:]))
        self.assertEqual(source[:start], candidate[len("#include <cstdlib>\n#include <fstream>\n"):
                                                   candidate.index(HELPER)])
        self.assertIn("static const int mode", candidate)
        self.assertIn("value == 0 || value == 1", candidate)
        self.assertIn('"MatMulNBitsGroupedRows4StaticWeightsF16" : "MatMulNBitsGroupedRows4F16"', candidate)

    def test_comparator_changes_only_profile_prefix(self):
        source = (SOURCE.parent / "native_compare.cpp").read_text()
        candidate = comparator_source(source)
        self.assertEqual(candidate.replace("row4-static-weights-v1-native-",
                                           "grouped-prefill-rows4-v2-native-"), source)
        self.assertNotIn("grouped-prefill-rows4-v2-native-", candidate)

    def test_static_register_and_dot_index_pairing(self):
        program = static_program(program_parts(SOURCE.read_text())[2])
        for word in range(4):
            self.assertIn(f"packed[{word}u]", program)
            self.assertIn(f"weights{word * 2} = vec4", program)
            self.assertIn(f"weights{word * 2 + 1} = vec4", program)
            self.assertIn(f'dot(tile_a[offset], weights{word * 2})', program)
            self.assertIn(f'dot(tile_a[offset + 1u], weights{word * 2 + 1})', program)


if __name__ == "__main__":
    unittest.main()
