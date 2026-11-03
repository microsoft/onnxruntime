import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from build_dense_packed_columns import DISPATCH, INCLUDES, SELECTOR, private_parts, transform_source, transform_template
from prepare_dense_packed_columns_proofs import base, model, screen


ROOT = Path(__file__).parent
REPOS = Path("/home/sayanshaw/repos")
SOURCE = REPOS / "qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc"
TEMPLATE = REPOS / "qwen38-int4-dense-block32/block32-template-v1/dense_block32_template.inc"


def render(source, sub_tile_count):
    parameter_type = 'onnxruntime::webgpu::wgsl_gen::TemplateParameter<"quantization/matmul_nbits.wgsl.template">::type'
    fields = list(dict.fromkeys(re.findall(r"params\.(param_\w+)", source)))
    booleans = {"acc_f32", "broadcast_a_row", "has_bias", "has_weight_idx",
                "has_weight_idx_indirect", "has_zero_points", "output_type_i32", "single_scale_weights"}
    declarations = "\n".join(f'{"bool" if name[6:] in booleans else "uint32_t"} {name} = 0;'
                             for name in fields)
    fixture = r'''
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>
#define ORT_RETURN_IF_NOT(condition, message) if (!(condition)) throw std::runtime_error(message);
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
  params.param_elements_in_value_b = 32;
  params.param_tile_size = 4; params.param_sub_tile_count = 2;
  params.param_tile_size_k = 1024; params.param_tile_size_k_vec = 32;
  params.param_a_length_per_tile = 256;
  try { FUNCTION(shader, params); }
  catch (const std::runtime_error& error) { std::cerr << error.what() << "\n"; return 2; }
  std::cout << shader.additional.str() << shader.main.str();
}
'''
    function = re.search(r"Status (\w+)\(", source).group(1)
    fixture = fixture.replace("params.param_sub_tile_count = 2;",
                              f"params.param_sub_tile_count = {sub_tile_count};")
    fixture = fixture.replace("FUNCTION(shader, params)", f"{function}(shader, params)")
    with tempfile.TemporaryDirectory(prefix="dense-packed-columns-render-") as directory:
        cpp, executable = Path(directory) / "render.cpp", Path(directory) / "render"
        cpp.write_text(fixture)
        subprocess.run(["c++", "-std=c++20", "-O0", str(cpp), "-o", str(executable)],
                       check=True, capture_output=True, text=True)
        return subprocess.check_output([str(executable)], text=True, stderr=subprocess.PIPE)


class DensePackedColumnsTests(unittest.TestCase):
    def test_mapping_activation_loads_and_tail_masks(self):
        for n in (1, 3, 4, 5, 320, 2560, 16384):
            for group in range((n + 3) // 4):
                cells = [(group * 4 + lane // 32 + component * 2, lane % 32)
                         for lane in range(64) for component in range(2)]
                self.assertEqual(sorted(cells),
                                 [(col, k) for col in range(group * 4, group * 4 + 4)
                                  for k in range(32)])
        self.assertEqual(sorted(id for lane in range(64) for id in range(lane, 256, 64)),
                         list(range(256)))
        for k in (640, 2560, 6144, 10240):
            for lane in range(32):
                self.assertEqual([tile // 32 + lane for tile in range(0, k, 1024)
                                  if tile // 32 + lane < k // 32],
                                 list(range(lane, k // 32, 32)))

    def test_real_emitter_preserves_decode_and_fold(self):
        original = render(TEMPLATE.read_text(), 4)
        candidate = render(transform_template(TEMPLATE.read_text()), 2)
        self.assertEqual(candidate[:candidate.index("  let idx =")].rstrip(),
                         original[:original.index("  let idx =")].rstrip())
        self.assertEqual(candidate[candidate.index("  if (batch >="):],
                         original[original.index("  if (batch >="):])
        self.assertEqual(candidate.count("workgroupBarrier();"), original.count("workgroupBarrier();"))
        self.assertIn("kidx += 1024u", candidate)
        self.assertIn("if (k_offset < uniforms.K_of_b)", candidate)
        self.assertIn("inter_results[idy + 2u][idx] = updated.y", candidate)
        self.assertIn("scale0 = scales[col0 * uniforms.blocks_per_col + block_idx + b_scale_offset]", candidate)
        self.assertIn("packed1 = b[col1 * uniforms.K_of_b + k_offset + b_base_offset]", candidate)
        self.assertEqual(candidate.count("let x0 = vec4<acc_element_t>(tile_A[a_offset]);"), 1)
        self.assertIn("var sum = vec2<f16>(0)", candidate)
        self.assertIn("a_offset += 2u", candidate)

    def test_reversible_private_kernel_and_unchanged_baselines(self):
        original = SOURCE.read_text()
        program, function = private_parts(original)
        candidate = transform_source(original)
        restored = candidate.replace(f'#include "{INCLUDES / "dense_packed_columns_template.inc"}"\n', "")
        restored = restored.replace("\n\n" + program, "", 1).replace("\n\n" + function, "", 1)
        restored = restored.replace(SELECTOR, "", 1).replace(DISPATCH, "", 1).replace(
            "(block32_mode == 1 ? (dense_packed_columns ? 4 : 3) : 1 + dense_tile4)",
            "(block32_mode == 1 ? 3 : 1 + dense_tile4)")
        self.assertEqual(restored, original)
        self.assertIn("constexpr uint32_t workgroup_size = 64;\n  constexpr uint32_t tile_size = 4;", function)
        self.assertIn("!context.EnableMatmulFp32Accumulation()", SELECTOR)
        sources = dict(
            native=base((ROOT / "run_dense_lane8_native.py").read_text()),
            model=model((ROOT / "run_dense_lane8_model.py").read_text()),
            screen=screen((ROOT / "run_hc_projection_packed16_screen.py").read_text()))
        for kind, expected in sources.items():
            actual = (ROOT / f"run_dense_packed_columns_{kind}.py").read_text()
            self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))
        model_source = (ROOT / "run_dense_packed_columns_model.py").read_text()
        self.assertIn("qwen38-int4-qmoe-prefill-rows4/build-v2", model_source)
        self.assertNotIn("qwen38-int4-qmoe-prefill-rows4/build-v1", model_source)
        self.assertIn('original_provider_control=args.mode == "control"', model_source)

    def test_canonical_guard_and_all_context_model_commands(self):
        with self.assertRaises(subprocess.CalledProcessError):
            render(transform_template(TEMPLATE.read_text()), 4)
        from run_dense_packed_columns_model import FLAG, TEMPLATE as MODEL_TEMPLATE, model_command, verify_profile

        template = json.loads((ROOT / f"{MODEL_TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "packed_columns"):
                    command = model_command(template, Path("/candidate/provider.so"),
                                            ROOT / "dense-packed-columns-command-test",
                                            context, full, mode, "standard")
                    self.assertIn(f"export {FLAG}={int(mode == 'packed_columns')}; ", command[-1])
                    self.assertIn(f"--contexts {context}", command[-1])
                    self.assertIn("--tokens 256 --repetitions 3", command[-1])
                    self.assertNotIn("--profile-decode-after", command[-1])
        counts = Counter(MatMulNBitsPackedColumnsF16=159, DynamicSparseAttentionQueryCache=12)
        verify_profile(counts, "packed_columns")
        counts["MatMulNBitsPackedColumnsF16"] = 158
        with self.assertRaises(AssertionError):
            verify_profile(counts, "packed_columns")


if __name__ == "__main__":
    unittest.main()
