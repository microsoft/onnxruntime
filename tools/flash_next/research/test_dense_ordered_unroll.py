import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from build_dense_ordered_unroll import transform_source, transform_template
from build_dense_flag_cache import compute_parts


REPOS = Path("/home/sayanshaw/repos")
SOURCE = REPOS / "qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc"
TEMPLATE = REPOS / "qwen38-int4-dense-block32/block32-template-v1/dense_block32_template.inc"


def render(source):
    parameter_type = 'onnxruntime::webgpu::wgsl_gen::TemplateParameter<"quantization/matmul_nbits.wgsl.template">::type'
    assert source.count(parameter_type) == 1
    fields = list(dict.fromkeys(re.findall(r"params\.(param_\w+)", source)))
    booleans = {"acc_f32", "broadcast_a_row", "has_bias", "has_weight_idx",
                "has_weight_idx_indirect", "has_zero_points", "output_type_i32", "single_scale_weights"}
    declarations = "\n".join(f'  {"bool" if name[6:] in booleans else "uint32_t"} {name} = 0;'
                             for name in fields)
    fixture = r'''
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
struct Status { static Status OK() { return {}; } };
struct Variable {
  std::string name;
  std::string GetByOffset(const std::string& index) { return name + "[" + index + "]"; }
  std::string SetByOffset(const std::string& index, const std::string& value) {
    return name + "[" + index + "] = " + value;
  }
};
struct ShaderHelper {
  std::ostringstream additional, main;
  auto& AdditionalImplementation() { return additional; }
  auto& MainFunctionBody() { return main; }
};
struct Params {
''' + declarations + r'''
  Variable *var_a, *var_b, *var_output, *var_scales_b;
};
''' + source.replace(parameter_type, "Params") + r'''
int main() {
  ShaderHelper shader;
  Params params;
  Variable a{"a"}, b{"b"}, output{"output"}, scales{"scales"};
  params.var_a = &a; params.var_b = &b; params.var_output = &output; params.var_scales_b = &scales;
  params.param_n_bits = 4;
  params.param_component_a = 4; params.param_component_b = 4;
  params.param_elements_in_value_b = 8;
  params.param_tile_size = 4; params.param_sub_tile_count = 4;
  params.param_tile_size_k = 512; params.param_tile_size_k_vec = 16;
  params.param_a_length_per_tile = 128;
  FUNCTION(shader, params);
  std::cout << shader.additional.str() << shader.main.str();
}
'''
    function = re.search(r"Status (\w+)\(", source).group(1)
    fixture = fixture.replace("FUNCTION(shader, params)", f"{function}(shader, params)")
    with tempfile.TemporaryDirectory(prefix="dense-unroll-render-") as directory:
        cpp, exe = Path(directory) / "render.cpp", Path(directory) / "render"
        cpp.write_text(fixture)
        subprocess.run(["c++", "-std=c++20", "-O0", str(cpp), "-o", str(exe)], check=True,
                       capture_output=True, text=True)
        return subprocess.check_output([str(exe)], text=True)


class DenseOrderedUnrollTests(unittest.TestCase):
    def test_original_compute_and_fallbacks_preserved(self):
        source = SOURCE.read_text()
        candidate = transform_source(source)
        self.assertEqual(compute_parts(source)[2], compute_parts(candidate)[2])
        start = source.index("Status ApplyDenseBlock32MatMulNBits(", source.index("Status ApplyDenseBlock32MatMulNBits(") + 1)
        self.assertEqual(source[start:], candidate[candidate.rindex(source[start:start + 80]):])
        self.assertIn('"MatMulNBitsBlock32F16Unroll2" : "MatMulNBitsBlock32F16"', candidate)
        self.assertIn("static const int mode", candidate)
        self.assertIn("value == 0 || value == 1", candidate)

    def test_real_emitter_preserves_both_tile_bodies(self):
        original = render(TEMPLATE.read_text())
        candidate = render(transform_template(TEMPLATE.read_text()))
        original_start = original.index("  for (var kidx =")
        original_body = original.index("    for (var id =", original_start)
        original_end = original.index("  }\n\n  if (batch >=", original_body)
        body = original[original_body:original_end]
        candidate_start = candidate.index("  for (var pair_kidx =")
        candidate_end = candidate.index("  }\n\n  if (batch >=", candidate_start)
        expected = ("  for (var pair_kidx = 0u; pair_kidx < uniforms.K; pair_kidx += 1024)\n"
                    "  {\n    {\n      let kidx = pair_kidx;\n" + body +
                    "    }\n    if (pair_kidx + 512 < uniforms.K) {\n"
                    "      let kidx = pair_kidx + 512;\n" + body + "    }\n")
        self.assertEqual(candidate[candidate_start:candidate_end], expected)
        self.assertEqual(original[:original_start], candidate[:candidate_start])
        self.assertEqual(original[original_end:], candidate[candidate_end:])
        self.assertEqual(body.count("workgroupBarrier();"), 2)

    def test_tile_order_and_uniform_tails(self):
        for tile in (32, 128, 512):
            for k in (0, 1, 31, 32, 128, 384, 512, 640, 1024, 2560, 6144, 10240):
                expected = list(range(0, k, tile))
                actual = [base + step * tile for base in range(0, k, 2 * tile)
                          for step in (0, 1) if base + step * tile < k]
                self.assertEqual(actual, expected, (k, tile))
                self.assertEqual(len(actual) * 2, len(expected) * 2)

    def test_model_commands_disable_statistics_during_timing(self):
        from run_dense_ordered_unroll_model import DIRECTORY, FLAG, ROOT, TEMPLATE, model_command

        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "unroll"):
                    command = model_command(template, provider, ROOT / "unroll-command-test",
                                            context, full, mode, "standard")
                    self.assertIn("--tokens 256 --repetitions 3", command[-1])
                    self.assertIn(f"--contexts {context}", command[-1])
                    self.assertIn(f"export {FLAG}={int(mode == 'unroll')}; ", command[-1])
                    self.assertIn("unset ORT_WEBGPU_DENSE_ORDERED_UNROLL2_STATS_FILE; ", command[-1])
                    self.assertNotIn("export ORT_WEBGPU_DENSE_ORDERED_UNROLL2_STATS_FILE=", command[-1])
                    self.assertNotIn("--profile-prefix", command[-1])
        coverage = model_command(template, provider, ROOT / "unroll-command-test",
                                 2048, True, "unroll", "coverage")
        self.assertIn("--tokens 36 --repetitions 1", coverage[-1])
        self.assertIn("export ORT_WEBGPU_DENSE_ORDERED_UNROLL2_STATS_FILE=", coverage[-1])


if __name__ == "__main__":
    unittest.main()
