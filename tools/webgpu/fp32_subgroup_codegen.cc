// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Offline template validation only: scalar bindings replace ShaderHelper's
// packed bindings. This never fabricates adapter capabilities or executes a GPU.
#include <array>
#include <fstream>
#include <sstream>
#include <string>

namespace offline {

struct Status {
  static Status OK() { return {}; }
};

struct ShaderVariableHelper {
  std::string name;
  std::string GetByOffset(const std::string& offset) const { return name + "[" + offset + "]"; }
  std::string SetByOffset(const std::string& offset, const std::string& value) const {
    return GetByOffset(offset) + " = " + value;
  }
};

struct ShaderHelper {
  std::ostringstream implementation;
  std::ostringstream body;
  auto& AdditionalImplementation() { return implementation; }
  auto& MainFunctionBody() { return body; }
};

template <size_t N>
struct string_template_filepath {
  std::array<char, N> data{};
  consteval string_template_filepath(const char (&value)[N + 1]) {
    for (size_t i = 0; i < N; ++i) data[i] = value[i];
  }
};
template <size_t N>
string_template_filepath(const char (&)[N]) -> string_template_filepath<N - 1>;
template <string_template_filepath Name>
struct TemplateParameter;
template <string_template_filepath Name>
Status ApplyTemplate(ShaderHelper&, typename TemplateParameter<Name>::type);

#define INCLUDED_BY_WGSL_GEN_HEADER
#include "wgsl_template_gen/index.h"
#undef INCLUDED_BY_WGSL_GEN_HEADER
#define INCLUDED_BY_WGSL_GEN_IMPL
#include "wgsl_template_gen/string_table.h"
#define MainFunctionStart() \
  {                         \
    auto& ss = shader_helper.MainFunctionBody();
#define MainFunctionEnd() }
#include "wgsl_template_gen/generated/quantization/subgroup_matrix_matmul_nbits_fp32.h"
#undef MainFunctionStart
#undef MainFunctionEnd
#undef INCLUDED_BY_WGSL_GEN_IMPL

}  // namespace offline

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const bool mixed = std::string{argv[1]} == "mixed";
  offline::ShaderHelper shader;
  const offline::ShaderVariableHelper a{"input_a"}, b{"input_b"}, scales{"scales_b"}, output{"output"};
  offline::ApplyTemplate<"quantization/subgroup_matrix_matmul_nbits_fp32.wgsl.template">(
      shader, {.param_has_bias = 1,
               .param_has_weight_idx = 0,
               .param_has_zero_points = 1,
               .param_n_bits = 4,
               .param_operand_f32 = !mixed,
               .param_output_type_i32 = 0,
               .param_sg_mat_k = mixed ? 16 : 8,
               .param_sg_mat_m = mixed ? 16 : 8,
               .param_sg_mat_n = mixed ? 16 : 8,
               .param_tile_size_k = 32,
               .param_tile_size_m = mixed ? 64 : 32,
               .param_tile_size_n = mixed ? 128 : 64,
               .var_a = &a,
               .var_b = &b,
               .var_output = &output,
               .var_scales_b = &scales});
  std::ofstream file{argv[2]};
  file << R"(enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);
alias output_element_t = f16;
struct Uniforms {
  M: u32, N: u32, K: u32, zero_blocks_per_col: u32,
  num_N_tile: u32, num_M_tile: u32, weight_idx: u32,
}
@group(0) @binding(0) var<storage, read> input_a: array<f16>;
@group(0) @binding(1) var<storage, read> input_b: array<u32>;
@group(0) @binding(2) var<storage, read> scales_b: array<f16>;
@group(0) @binding(3) var<storage, read> zero_points: array<u32>;
@group(0) @binding(4) var<storage, read> bias: array<f16>;
@group(0) @binding(5) var<storage, read_write> output: array<f16>;
@group(0) @binding(6) var<uniform> uniforms: Uniforms;
const workgroup_size_x = 128u;
)";
  file << shader.implementation.str();
  file << R"(@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32,
        @builtin(subgroup_size) sg_size: u32) {
)";
  file << shader.body.str() << "}\n";
  return file ? 0 : 1;
}
