// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <string_view>

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace {


class DenseTileParallel3PackedProgram final : public Program<DenseTileParallel3PackedProgram> {
 public:
  DenseTileParallel3PackedProgram() : Program{"DenseTileParallel3Packed"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"blocks", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << R"(
var<workgroup> tile_a: array<vec4<f16>, 1536>;
var<workgroup> tile_sums: array<u32, 768>;
fn decode_low_nibbles(word: u32) -> vec4<f16> {
  let bottom = bitcast<vec2<f16>>((word & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);
  let top = bitcast<vec2<f16>>(((word >> 8u) & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);
  return vec4<f16>(bottom.x, top.x, bottom.y, top.y);
}
)";
    auto& code = shader.MainFunctionBody();
    code << R"(
let logical_idx = local_idx % 128u;
let tile_worker = local_idx / 128u;
let stream = logical_idx % 32u;
let column_group = logical_idx / 32u;
for (var wave = 0u; wave * 96u < uniforms.blocks; wave++) {
  let tile = wave * 3u + tile_worker;
  let buffer = (wave % 2u) * 3u + tile_worker;
  for (var i = logical_idx; i < 256u; i += 128u) {
    var value = vec4<f16>(0);
    let block = tile * 32u + i / 8u;
    if (block < uniforms.blocks) {
)" << "      value = " << a.GetByOffset("block * 8u + i % 8u") << ";\n"
         << R"(
    }
    tile_a[buffer * 256u + i] = value;
  }
  workgroupBarrier();
  let block = tile * 32u + stream;
  var sum0 = f16(0);
  var sum1 = f16(0);
  if (block < uniforms.blocks) {
)";

    code << "    let col0 = workgroup_idx * 8u + column_group + 0u;\n"
         << "    let packed0 = " << b.GetByOffset("col0 * uniforms.blocks + block") << ";\n"
         << "    let scale0 = " << scales.GetByOffset("col0 * uniforms.blocks + block") << ";\n";

    code << "    let col1 = workgroup_idx * 8u + column_group + 4u;\n"
         << "    let packed1 = " << b.GetByOffset("col1 * uniforms.blocks + block") << ";\n"
         << "    let scale1 = " << scales.GetByOffset("col1 * uniforms.blocks + block") << ";\n";

    code << R"(
    for (var word = 0u; word < 4u; word++) {
      let x0 = tile_a[buffer * 256u + stream * 8u + word * 2u];
      let x1 = tile_a[buffer * 256u + stream * 8u + word * 2u + 1u];
)";

    code << "      {\n"
         << "        let low = decode_low_nibbles(packed0[word]);\n"
         << "        let high = decode_low_nibbles(packed0[word] >> 4u);\n"
         << "        let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale0;\n"
         << "        let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale0;\n"
         << "        sum0 += dot(x0, w0) + dot(x1, w1);\n"
         << "      }\n";

    code << "      {\n"
         << "        let low = decode_low_nibbles(packed1[word]);\n"
         << "        let high = decode_low_nibbles(packed1[word] >> 4u);\n"
         << "        let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale1;\n"
         << "        let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale1;\n"
         << "        sum1 += dot(x0, w0) + dot(x1, w1);\n"
         << "      }\n";
    code << "    }\n";

    code << R"(
  }
  tile_sums[tile * 128u + logical_idx] = bitcast<u32>(vec2<f16>(sum0, sum1));
}
workgroupBarrier();
if (local_idx < 128u) {
  var partial0 = f16(0);
  var partial1 = f16(0);
  for (var tile = 0u; tile * 32u < uniforms.blocks; tile++) {
    if (tile * 32u + stream < uniforms.blocks) {
      let pair = bitcast<vec2<f16>>(tile_sums[tile * 128u + logical_idx]);
      partial0 += pair.x;
      partial1 += pair.y;
    }
  }
  tile_sums[logical_idx] = bitcast<u32>(vec2<f16>(partial0, partial1));
}
workgroupBarrier();
if (local_idx < 8u) {
  var value = f16(0);
  for (var stream = 0u; stream < 32u; stream++) {
    let pair = bitcast<vec2<f16>>(tile_sums[(local_idx % 4u) * 32u + stream]);
    value += pair[local_idx / 4u];
  }
)" << y.SetByOffset("workgroup_idx * 8u + local_idx", "value") << "\n}\n";
    return Status::OK();
  }
};


class DenseColumns8DoubleBufferProgram final : public Program<DenseColumns8DoubleBufferProgram> {
 public:
  DenseColumns8DoubleBufferProgram() : Program{"DenseColumns8DoubleBuffer"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"blocks", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << R"(
var<workgroup> tile_a: array<vec4<f16>, 512>;
var<workgroup> partials: array<f16, 256>;
fn decode_low_nibbles(word: u32) -> vec4<f16> {
  let bottom = bitcast<vec2<f16>>((word & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);
  let top = bitcast<vec2<f16>>(((word >> 8u) & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);
  return vec4<f16>(bottom.x, top.x, bottom.y, top.y);
}
)";
    auto& code = shader.MainFunctionBody();
    code << R"(
let stream = local_idx % 32u;
let column_group = local_idx / 32u;
for (var i = local_idx; i < 256u; i += 128u) { partials[i] = f16(0); }
for (var tile = 0u; tile * 32u < uniforms.blocks; tile++) {
  for (var i = local_idx; i < 256u; i += 128u) {
    var value = vec4<f16>(0);
    let block = tile * 32u + i / 8u;
    if (block < uniforms.blocks) {
)" << "      value = " << a.GetByOffset("block * 8u + i % 8u") << ";\n"
         << R"(
    }
    tile_a[(tile % 2u) * 256u + i] = value;
  }
  workgroupBarrier();
  let block = tile * 32u + stream;
  if (block < uniforms.blocks) {
)";

    code << "    let col0 = workgroup_idx * 8u + column_group + 0u;\n"
         << "    let packed0 = " << b.GetByOffset("col0 * uniforms.blocks + block") << ";\n"
         << "    let scale0 = " << scales.GetByOffset("col0 * uniforms.blocks + block") << ";\n"
         << "    var sum0 = f16(0);\n";

    code << "    let col1 = workgroup_idx * 8u + column_group + 4u;\n"
         << "    let packed1 = " << b.GetByOffset("col1 * uniforms.blocks + block") << ";\n"
         << "    let scale1 = " << scales.GetByOffset("col1 * uniforms.blocks + block") << ";\n"
         << "    var sum1 = f16(0);\n";

    code << R"(
    for (var word = 0u; word < 4u; word++) {
      let x0 = tile_a[(tile % 2u) * 256u + stream * 8u + word * 2u];
      let x1 = tile_a[(tile % 2u) * 256u + stream * 8u + word * 2u + 1u];
)";

    code << "      {\n"
         << "        let low = decode_low_nibbles(packed0[word]);\n"
         << "        let high = decode_low_nibbles(packed0[word] >> 4u);\n"
         << "        let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale0;\n"
         << "        let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale0;\n"
         << "        sum0 += dot(x0, w0) + dot(x1, w1);\n"
         << "      }\n";

    code << "      {\n"
         << "        let low = decode_low_nibbles(packed1[word]);\n"
         << "        let high = decode_low_nibbles(packed1[word] >> 4u);\n"
         << "        let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale1;\n"
         << "        let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale1;\n"
         << "        sum1 += dot(x0, w0) + dot(x1, w1);\n"
         << "      }\n";
    code << "    }\n";
    code << "    partials[(column_group + 0u) * 32u + stream] += sum0;\n";
    code << "    partials[(column_group + 4u) * 32u + stream] += sum1;\n";

    code << R"(
  }
}
workgroupBarrier();
if (local_idx < 8u) {
  var value = f16(0);
  for (var stream = 0u; stream < 32u; stream++) {
    value += partials[local_idx * 32u + stream];
  }
)" << y.SetByOffset("workgroup_idx * 8u + local_idx", "value") << "\n}\n";
    return Status::OK();
  }
};

class GroupedPrefillRows4Program final : public Program<GroupedPrefillRows4Program> {
 public:
  explicit GroupedPrefillRows4Program(bool bias)
      : Program{"MatMulNBitsGroupedRows4F16"}, bias_{bias} {}

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32});

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    if (bias_) shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    const auto& experts = shader.AddInput("experts", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << R"(
var<workgroup> tile_a: array<vec4<f16>, 1024>;
var<workgroup> partials: array<array<array<f16, 32>, 8>, 4>;
var<workgroup> row_experts: array<u32, 4>;
)";
    auto& code = shader.MainFunctionBody();
    code << "let first_row = (workgroup_idx / uniforms.num_N_tile) * 4u;\n"
         << "let first_col = (workgroup_idx % uniforms.num_N_tile) * 8u;\n"
         << "let idx = local_idx % 32u;\n"
         << "let idy = local_idx / 32u;\n"
         << "if (local_idx < 4u) { row_experts[local_idx] = "
         << experts.GetByOffset("first_row + local_idx") << "; }\n"
         << "for (var kidx = 0u; kidx < uniforms.K; kidx += 1024u) {\n"
         << "  for (var i = local_idx; i < 1024u; i += 128u) {\n"
         << "    let row = i / 256u;\n"
         << "    let k = kidx / 4u + i % 256u;\n"
         << "    if (k < uniforms.K / 4u) { tile_a[i] = "
         << a.GetByOffset("(first_row + row) * (uniforms.K / 4u) + k") << "; }\n"
         << "    else { tile_a[i] = vec4<f16>(0); }\n"
         << "  }\n"
         << "  workgroupBarrier();\n"
         << "  for (var slot = 0u; slot < 8u; slot += 4u) {\n"
         << "    let col = first_col + slot + idy;\n"
         << "    let block = kidx / 32u + idx;\n"
         << "    if (block < uniforms.K / 32u) {\n"
         << "      var cached_expert = 0xffffffffu;\n"
         << "      var weights: array<vec4<f16>, 8>;\n"
         << "      for (var row = 0u; row < 4u; row++) {\n"
         << "        let expert = row_experts[row];\n"
         << "        if (expert != cached_expert) {\n"
         << "          let scale = "
         << scales.GetByOffset("(expert * uniforms.N + col) * (uniforms.K / 32u) + block")
         << ";\n"
         << "          let packed = "
         << b.GetByOffset("(expert * uniforms.N + col) * (uniforms.K / 32u) + block") << ";\n"
         << "          for (var word = 0u; word < 4u; word++) {\n"
         << "            let lo = vec4<f16>(unpack4xU8(packed[word] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "            let hi = vec4<f16>(unpack4xU8((packed[word] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "            weights[word * 2u] = vec4<f16>(lo.x, hi.x, lo.y, hi.y) * scale;\n"
         << "            weights[word * 2u + 1u] = vec4<f16>(lo.z, hi.z, lo.w, hi.w) * scale;\n"
         << "          }\n"
         << "          cached_expert = expert;\n"
         << "        }\n"
         << "        var sum = f16(0);\n"
         << "        for (var word = 0u; word < 4u; word++) {\n"
         << "          let offset = row * 256u + idx * 8u + word * 2u;\n"
         << "          sum += dot(tile_a[offset], weights[word * 2u]) +\n"
         << "                 dot(tile_a[offset + 1u], weights[word * 2u + 1u]);\n"
         << "        }\n"
         << "        partials[row][slot + idy][idx] += sum;\n"
         << "      }\n"
         << "    }\n"
         << "  }\n"
         << "  workgroupBarrier();\n"
         << "}\n"
         << "if (local_idx < 8u) {\n"
         << "  let col = first_col + local_idx;\n"
         << "  for (var row = 0u; row < 4u; row++) {\n"
         << "    var value = f16(0);\n"
         << "    for (var lane = 0u; lane < 32u; lane++) {\n"
         << "      value += partials[row][local_idx][lane];\n"
         << "    }\n";
    if (bias_) code << "    value += bias[row_experts[row] * uniforms.N + col];\n";
    code << "    " << y.SetByOffset("(first_row + row) * uniforms.N + col", "value") << "\n"
         << "  }\n"
         << "}\n";
    return Status::OK();
  }

 private:
  bool bias_;
};

class MatMulInt3Program final : public Program<MatMulInt3Program> {
 public:
  MatMulInt3Program(bool indirect, bool bias)
      : Program{"MatMulInt3"}, indirect_{indirect}, bias_{bias} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    if (indirect_) shader.AddInput("experts", ShaderUsage::UseValueTypeAlias);
    if (bias_) shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << "var<workgroup> partials: array<f32, 128>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let row = workgroup_idx / uniforms.N;\n"
         << "let col = workgroup_idx % uniforms.N;\n"
         << "let input_row = select(row, 0u, uniforms.broadcast_a != 0u);\n";
    code << (indirect_ ? "let expert = experts[row];\n" : "let expert = uniforms.weight_index;\n");
    code << "let packed_row = (expert * uniforms.N + col) * (uniforms.K * 3u / 32u);\n"
         << "let scale_row = (expert * uniforms.N + col) * (uniforms.K / 32u);\n"
         << "var sum = 0.0f;\n"
         << "for (var k = local_idx; k < uniforms.K; k += 128u) {\n"
         << "  let bit = k * 3u;\n"
         << "  let shift = bit % 32u;\n"
         << "  let word = packed_row + bit / 32u;\n"
         << "  var q = " << b.GetByOffset("word") << " >> shift;\n"
         << "  if (shift > 29u) { q |= " << b.GetByOffset("word + 1u") << " << (32u - shift); }\n"
         << "  let weight = scales_element_t(i32(q & 7u) - 4) * "
         << scales.GetByOffset("scale_row + k / 32u") << ";\n"
         << "  sum += f32(" << a.GetByOffset("input_row * uniforms.K + k") << ") * f32(weight);\n"
         << "}\n"
         << "partials[local_idx] = sum;\n"
         << "workgroupBarrier();\n"
         << "for (var stride = 64u; stride > 0u; stride /= 2u) {\n"
         << "  if (local_idx < stride) { partials[local_idx] += partials[local_idx + stride]; }\n"
         << "  workgroupBarrier();\n"
         << "}\n"
         << "if (local_idx == 0u) {\n"
         << "  var result = y_element_t(partials[0]);\n";
    if (bias_) code << "  result += bias[expert * uniforms.N + col];\n";
    code << y.SetByOffset("row * uniforms.N + col", "result") << "\n}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"weight_index", ProgramUniformVariableDataType::Uint32},
      {"broadcast_a", ProgramUniformVariableDataType::Uint32});

 private:
  bool indirect_;
  bool bias_;
};

class ExactDenseMidProgram final : public Program<ExactDenseMidProgram> {
 public:
  ExactDenseMidProgram() : Program{"ExactDenseMid"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << "var<workgroup> partials: array<f16, 64>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let lane = local_idx % 32u;\n"
         << "let col = workgroup_idx * 2u + local_idx / 32u;\n"
         << "var total = f16(0);\n"
         << "for (var tile = 0u; tile < 3u; tile++) {\n"
         << "  let block = tile * 32u + lane;\n"
         << "  if (block < 80u) {\n"
         << "    let packed = " << b.GetByOffset("col * 80u + block") << ";\n"
         << "    let scale = " << scales.GetByOffset("col * 80u + block") << ";\n"
         << "    var sum = f16(0);\n"
         << "    for (var i = 0u; i < 4u; i++) {\n"
         << "      let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "      let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "      let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "      let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "      let x0 = " << a.GetByOffset("block * 8u + i * 2u") << ";\n"
         << "      let x1 = " << a.GetByOffset("block * 8u + i * 2u + 1u") << ";\n"
         << "      sum += dot(x0, w0) + dot(x1, w1);\n"
         << "    }\n"
         << "    total += sum;\n"
         << "  }\n"
         << "}\n"
         << "partials[local_idx] = total;\n"
         << "workgroupBarrier();\n"
         << "if (lane == 0u) {\n"
         << "  var value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\n"
         << y.SetByOffset("col", "value") << "\n"
         << "}\n";
    return Status::OK();
  }
};

class HCExactDenseProgram final : public Program<HCExactDenseProgram> {
 public:
  HCExactDenseProgram() : Program{"HCExactDense"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << "var<workgroup> partials: array<f16, 64>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let lane = local_idx % 32u;\n"
         << "let col = workgroup_idx * 2u + local_idx / 32u;\n"
         << "var total = f16(0);\n"
         << "for (var tile = 0u; tile < 10u; tile++) {\n"
         << "  let block = tile * 32u + lane;\n"
         << "  let packed = " << b.GetByOffset("col * 320u + block") << ";\n"
         << "  let scale = " << scales.GetByOffset("col * 320u + block") << ";\n"
         << "  var sum = f16(0);\n"
         << "  for (var i = 0u; i < 4u; i++) {\n"
         << "    let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "    let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "    let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "    let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "    let x0 = " << a.GetByOffset("block * 8u + i * 2u") << ";\n"
         << "    let x1 = " << a.GetByOffset("block * 8u + i * 2u + 1u") << ";\n"
         << "    sum += dot(x0, w0) + dot(x1, w1);\n"
         << "  }\n"
         << "  total += sum;\n"
         << "}\n"
         << "partials[local_idx] = total;\n"
         << "workgroupBarrier();\n"
         << "if (lane == 0u) {\n"
         << "  var value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\n"
         << y.SetByOffset("col", "value") << "\n"
         << "}\n";
    return Status::OK();
  }
};

class MatMulNBitsSkinnyProgram final : public Program<MatMulNBitsSkinnyProgram> {
 public:
  MatMulNBitsSkinnyProgram() : Program{"MatMulNBitsSkinny"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << "var<workgroup> partials: array<f32, 128>;\n";
    shader.MainFunctionBody()
        << "let row = workgroup_idx / uniforms.N;\n"
        << "let col = workgroup_idx % uniforms.N;\n"
        << "if (row >= uniforms.M) { return; }\n"
        << "var sum = 0.0f;\n"
        << "for (var word = local_idx; word < uniforms.K / 8u; word += 128u) {\n"
        << "  let packed = " << b.GetByOffset("col * (uniforms.K / 8u) + word") << ";\n"
        << "  let scale = " << scales.GetByOffset("col * (uniforms.K / 32u) + word / 4u") << ";\n"
        << "  let low = vec4<f16>(unpack4xU8(packed & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
        << "  let high = vec4<f16>(unpack4xU8((packed >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
        << "  let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
        << "  let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
        << "  let x0 = " << a.GetByOffset("row * (uniforms.K / 4u) + word * 2u") << ";\n"
        << "  let x1 = " << a.GetByOffset("row * (uniforms.K / 4u) + word * 2u + 1u") << ";\n"
        << "  sum += dot(vec4<f32>(x0), vec4<f32>(w0)) + dot(vec4<f32>(x1), vec4<f32>(w1));\n"
        << "}\n"
        << "partials[local_idx] = sum;\n"
        << "workgroupBarrier();\n"
        << "for (var stride = 64u; stride > 0u; stride /= 2u) {\n"
        << "  if (local_idx < stride) { partials[local_idx] += partials[local_idx + stride]; }\n"
        << "  workgroupBarrier();\n"
        << "}\n"
        << "if (local_idx == 0u) {\n"
        << y.SetByOffset("row * uniforms.N + col", "y_element_t(partials[0])") << "\n"
        << "}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32});
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulNBits);

Status MatMulNBitsWideTileProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& b = shader.AddInput("input_b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  }
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  const uint32_t workgroup_size = WorkgroupSizeX() * WorkgroupSizeY() * WorkgroupSizeZ();
  ORT_ENFORCE(tile_n_ == workgroup_size, "tile_n must be workgroup_size.");
  ORT_ENFORCE(tile_m_ % (workgroup_size / 8) == 0, "tile_m must be a multiple of workgroup_size / 8.");
  ORT_ENFORCE(nbits_ == 4 || nbits_ == 8, "Only 4/8 bits are supported for webgpu matmulnbits.");

  return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_wide_tile.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(acc_f32, acc_f32_),
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                             WGSL_TEMPLATE_PARAMETER(nbits, nbits_),
                             WGSL_TEMPLATE_PARAMETER(subgroup_min_size, subgroup_min_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_m, tile_m_),
                             WGSL_TEMPLATE_PARAMETER(tile_n, tile_n_),
                             WGSL_TEMPLATE_VARIABLE(a, a),
                             WGSL_TEMPLATE_VARIABLE(b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales, scales));
}

// Apply similar idea with DP4AMatMulNBitsSmallMProgram algorithm.
namespace {
#include "dense_bitcast_template.inc"
#include "dense_block32_template.inc"
class DenseBitcastProgram final : public Program<DenseBitcastProgram> {
 public:
  DenseBitcastProgram(uint32_t tile_size, uint32_t nbits, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect, bool single_scale_weights, uint32_t tile_size_k_vec = 16, bool broadcast_a_row = false, bool acc_f32 = false)
      : Program{"MatMulNBitsBitcastF16"}, tile_size_(tile_size), nbits_(nbits), has_zero_points_(has_zero_points), has_bias_(has_bias), has_weight_idx_{has_weight_idx}, has_weight_idx_indirect_{has_weight_idx_indirect}, single_scale_weights_(single_scale_weights), tile_size_k_vec_(tile_size_k_vec), broadcast_a_row_(broadcast_a_row), acc_f32_(acc_f32) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"K_of_a", ProgramUniformVariableDataType::Uint32},
      {"K_of_b", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"zero_blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32},
      {"batch_count", ProgramUniformVariableDataType::Uint32},
      {"weight_idx", ProgramUniformVariableDataType::Uint32},
      {"dispatch_M", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_;
  uint32_t nbits_;
  bool has_zero_points_;
  bool has_bias_;
  bool has_weight_idx_;
  bool has_weight_idx_indirect_;
  bool single_scale_weights_;
  uint32_t tile_size_k_vec_;
  bool broadcast_a_row_;
  bool acc_f32_;
};

Status DenseBitcastProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b");
  const auto& scales_b = shader.AddInput("scales_b");
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

  const uint32_t components_a = a.NumComponents();
  const uint32_t components_b = b.NumComponents() / 4;  // b is stored as uint32 which includes 4 uint8.
  const uint32_t tile_size_k_vec = tile_size_k_vec_;
  const uint32_t elements_in_value_b = components_b * (32 / nbits_);
  const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
  const uint32_t a_length_per_tile = tile_size_k / components_a;
  uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;

  return ApplyDenseBitcastTemplate(shader, {
                             WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                             WGSL_TEMPLATE_PARAMETER(acc_f32, acc_f32_),
                             WGSL_TEMPLATE_PARAMETER(broadcast_a_row, broadcast_a_row_),
                             WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                             WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                             WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits_),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                             WGSL_TEMPLATE_VARIABLE(a, a),
                             WGSL_TEMPLATE_VARIABLE(b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b)});
}

Status ApplyDenseBitcastMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points, const Tensor* bias,
                        int64_t K_op, int64_t N_op, int64_t block_size_op, int64_t accuracy_level, int64_t bits_op,
                        onnxruntime::webgpu::ComputeContext& context, Tensor* y, const uint32_t weight_index = 0,
                        const Tensor* weight_index_indirect = nullptr,
                        uint32_t override_M = 0);


class DenseTile4Program final : public Program<DenseTile4Program> {
 public:
  DenseTile4Program(uint32_t tile_size, uint32_t nbits, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect, bool single_scale_weights, uint32_t tile_size_k_vec = 16, bool broadcast_a_row = false, bool acc_f32 = false)
      : Program{"MatMulNBitsTile4F16"}, tile_size_(tile_size), nbits_(nbits), has_zero_points_(has_zero_points), has_bias_(has_bias), has_weight_idx_{has_weight_idx}, has_weight_idx_indirect_{has_weight_idx_indirect}, single_scale_weights_(single_scale_weights), tile_size_k_vec_(tile_size_k_vec), broadcast_a_row_(broadcast_a_row), acc_f32_(acc_f32) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"K_of_a", ProgramUniformVariableDataType::Uint32},
      {"K_of_b", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"zero_blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32},
      {"batch_count", ProgramUniformVariableDataType::Uint32},
      {"weight_idx", ProgramUniformVariableDataType::Uint32},
      {"dispatch_M", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_;
  uint32_t nbits_;
  bool has_zero_points_;
  bool has_bias_;
  bool has_weight_idx_;
  bool has_weight_idx_indirect_;
  bool single_scale_weights_;
  uint32_t tile_size_k_vec_;
  bool broadcast_a_row_;
  bool acc_f32_;
};

Status DenseTile4Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b");
  const auto& scales_b = shader.AddInput("scales_b");
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

  const uint32_t components_a = a.NumComponents();
  const uint32_t components_b = b.NumComponents() / 4;  // b is stored as uint32 which includes 4 uint8.
  const uint32_t tile_size_k_vec = tile_size_k_vec_;
  const uint32_t elements_in_value_b = components_b * (32 / nbits_);
  const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
  const uint32_t a_length_per_tile = tile_size_k / components_a;
  uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;

  return ApplyDenseBitcastTemplate(shader, {
                             WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                             WGSL_TEMPLATE_PARAMETER(acc_f32, acc_f32_),
                             WGSL_TEMPLATE_PARAMETER(broadcast_a_row, broadcast_a_row_),
                             WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                             WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                             WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits_),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                             WGSL_TEMPLATE_VARIABLE(a, a),
                             WGSL_TEMPLATE_VARIABLE(b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b)});
}

Status ApplyDenseTile4MatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points, const Tensor* bias,
                        int64_t K_op, int64_t N_op, int64_t block_size_op, int64_t accuracy_level, int64_t bits_op,
                        onnxruntime::webgpu::ComputeContext& context, Tensor* y, const uint32_t weight_index = 0,
                        const Tensor* weight_index_indirect = nullptr,
                        uint32_t override_M = 0);


class DenseBlock32Program final : public Program<DenseBlock32Program> {
 public:
  DenseBlock32Program(uint32_t tile_size, uint32_t nbits, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect, bool single_scale_weights, uint32_t tile_size_k_vec = 16, bool broadcast_a_row = false, bool acc_f32 = false)
      : Program{"MatMulNBitsBlock32F16"}, tile_size_(tile_size), nbits_(nbits), has_zero_points_(has_zero_points), has_bias_(has_bias), has_weight_idx_{has_weight_idx}, has_weight_idx_indirect_{has_weight_idx_indirect}, single_scale_weights_(single_scale_weights), tile_size_k_vec_(tile_size_k_vec), broadcast_a_row_(broadcast_a_row), acc_f32_(acc_f32) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"K_of_a", ProgramUniformVariableDataType::Uint32},
      {"K_of_b", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"zero_blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32},
      {"batch_count", ProgramUniformVariableDataType::Uint32},
      {"weight_idx", ProgramUniformVariableDataType::Uint32},
      {"dispatch_M", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_;
  uint32_t nbits_;
  bool has_zero_points_;
  bool has_bias_;
  bool has_weight_idx_;
  bool has_weight_idx_indirect_;
  bool single_scale_weights_;
  uint32_t tile_size_k_vec_;
  bool broadcast_a_row_;
  bool acc_f32_;
};

Status DenseBlock32Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b");
  const auto& scales_b = shader.AddInput("scales_b");
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

  const uint32_t components_a = a.NumComponents();
  const uint32_t components_b = b.NumComponents() / 4;  // b is stored as uint32 which includes 4 uint8.
  const uint32_t tile_size_k_vec = tile_size_k_vec_;
  const uint32_t elements_in_value_b = components_b * (32 / nbits_);
  const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
  const uint32_t a_length_per_tile = tile_size_k / components_a;
  uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;

  return ApplyDenseBlock32Template(shader, {
                             WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                             WGSL_TEMPLATE_PARAMETER(acc_f32, acc_f32_),
                             WGSL_TEMPLATE_PARAMETER(broadcast_a_row, broadcast_a_row_),
                             WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                             WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                             WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits_),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                             WGSL_TEMPLATE_VARIABLE(a, a),
                             WGSL_TEMPLATE_VARIABLE(b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b)});
}

Status ApplyDenseBlock32MatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points, const Tensor* bias,
                        int64_t K_op, int64_t N_op, int64_t block_size_op, int64_t accuracy_level, int64_t bits_op,
                        onnxruntime::webgpu::ComputeContext& context, Tensor* y, const uint32_t weight_index = 0,
                        const Tensor* weight_index_indirect = nullptr,
                        uint32_t override_M = 0);

}

Status MatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b");
  const auto& scales_b = shader.AddInput("scales_b");
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

  const uint32_t components_a = a.NumComponents();
  const uint32_t components_b = b.NumComponents() / 4;  // b is stored as uint32 which includes 4 uint8.
  const uint32_t tile_size_k_vec = tile_size_k_vec_;
  const uint32_t elements_in_value_b = components_b * (32 / nbits_);
  const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
  const uint32_t a_length_per_tile = tile_size_k / components_a;
  uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;

  return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                             WGSL_TEMPLATE_PARAMETER(acc_f32, acc_f32_),
                             WGSL_TEMPLATE_PARAMETER(broadcast_a_row, broadcast_a_row_),
                             WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                             WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                             WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits_),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                             WGSL_TEMPLATE_VARIABLE(a, a),
                             WGSL_TEMPLATE_VARIABLE(b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));
}

Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  static const int tile_parallel3 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_DENSE_TILE_PARALLEL3_LONG_K", 0);
  ORT_RETURN_IF_NOT(tile_parallel3 == 0 || tile_parallel3 == 1, "ORT_WEBGPU_DENSE_TILE_PARALLEL3_LONG_K must be 0 or 1");
  static const int columns8_mode = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER", 0);
  ORT_RETURN_IF_NOT(columns8_mode == 0 || columns8_mode == 1, "ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER must be 0 or 1");

  const int block32_mode = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_DENSE_BLOCK32", 0);
  ORT_RETURN_IF_NOT(block32_mode == 0 || block32_mode == 1, "ORT_WEBGPU_DENSE_BLOCK32 must be 0 or 1");

  const int dense_tile4 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_DENSE_TILE4", 0);
  ORT_RETURN_IF_NOT(dense_tile4 == 0 || dense_tile4 == 1, "ORT_WEBGPU_DENSE_TILE4 must be 0 or 1");
  const int dense_bitcast = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_DENSE_BITCAST_UNPACK", 0);
  ORT_RETURN_IF_NOT(dense_bitcast == 0 || dense_bitcast == 1, "ORT_WEBGPU_DENSE_BITCAST_UNPACK must be 0 or 1");
  const Tensor* a = context.Input(0);
  const Tensor* b = context.Input(1);
  const Tensor* scales = context.Input(2);
  const Tensor* zero_points = context.Input(3);
  const Tensor* g_idx = context.Input(4);
  const Tensor* bias = context.Input(5);

  ORT_ENFORCE(g_idx == nullptr, "group_idx as input is not supported yet.");

  const bool has_zero_points = zero_points != nullptr;
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(), "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
  }

  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
  auto output_shape = helper.OutputShape();
  Tensor* y = context.Output(0, output_shape);
  const uint32_t data_size = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }


  const bool dense_bitcast_route =
      dense_bitcast == 1 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      bits_ == 4 && block_size_ == 32 && !has_zero_points && bias == nullptr &&
      !context.EnableMatmulFp32Accumulation() &&
      a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
      a->Shape().Size() == K_ && K_ % 32 == 0 &&
      ((K_ == 2560 && (N_ == 1 || N_ == 16384 || N_ == 7168 || N_ == 6144 ||
                      N_ == 10240 || N_ == 2560)) ||
       ((K_ == 6144 || K_ == 640) && N_ == 2560) ||
       (K_ == 10240 && N_ == 320));

  const bool columns8 = columns8_mode == 1 && dense_bitcast_route && block32_mode == 1 &&
      N_ >= 2560 && N_ % 8 == 0 && b->IsDataType<uint8_t>() &&
      b->Shape().Size() == N_ * (K_ / 2) && scales->Shape().Size() == N_ * (K_ / 32) &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= 4608;
  const bool use_tile_parallel3 = tile_parallel3 == 1 && columns8 &&
      N_ == 2560 && K_ == 6144 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      context.DeviceLimits().maxComputeWorkgroupSizeX >= 384 &&
      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 384 &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= 15360;
  auto compute = [&]() {
    if (use_tile_parallel3) {
      DenseTileParallel3PackedProgram program;
      program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
          .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
          .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank, 1})
          .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, 1})
          .AddUniformVariables({onnxruntime::narrow<uint32_t>(K_ / 32)})
          .SetWorkgroupSize(384).SetDispatchGroupSize(N_ / 8);
      return context.RunProgram(program);
    }
    if (columns8) {
      DenseColumns8DoubleBufferProgram program;
      program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
          .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
          .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank, 1})
          .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, 1})
          .AddUniformVariables({onnxruntime::narrow<uint32_t>(K_ / 32)})
          .SetWorkgroupSize(128).SetDispatchGroupSize(N_ / 8);
      return context.RunProgram(program);
    }
    if (dense_bitcast_route && block32_mode == 1) {
      return ApplyDenseBlock32MatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
    if (dense_bitcast_route && dense_tile4 == 1) {
      return ApplyDenseTile4MatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
    if (dense_bitcast_route) {
      return ApplyDenseBitcastMatMulNBits(a, b, scales, zero_points, bias,
          K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
    }
    return ApplyMatMulNBits(a, b, scales, zero_points, bias, K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
  };
  const bool exact_route =
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      bits_ == 4 && block_size_ == 32 && !has_zero_points && bias == nullptr &&
      !context.EnableMatmulFp32Accumulation() &&
      a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
      ((K_ == 2560 && (N_ == 512 || N_ == 640) &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXACT_DENSE_MID", 0) != 0) ||
       (K_ == 10240 && N_ == 324 &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_EXACT_DENSE", 0) != 0));

  const int generic_replay =
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_GENERIC_MATMUL_REPLAY", 0);
  ORT_RETURN_IF_NOT(generic_replay == 0 || generic_replay == 1,
                    "ORT_WEBGPU_GENERIC_MATMUL_REPLAY must be 0 or 1");
  const bool generic_shape =
      (K_ == 2560 && (N_ == 1 || N_ == 16384 || N_ == 7168 || N_ == 6144 ||
                     N_ == 10240 || N_ == 2560)) ||
      ((K_ == 6144 || K_ == 640) && N_ == 2560) ||
      (K_ == 10240 && N_ == 320);
  const bool generic_route =
      generic_replay == 1 && generic_shape &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      bits_ == 4 && block_size_ == 32 && !has_zero_points && bias == nullptr &&
      !context.EnableMatmulFp32Accumulation() &&
      a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_SKINNY", 0) == 0 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY", 0) == 0;
  if (!exact_route && !generic_route) return compute();

  return static_replay_.Run(context, {y}, compute, 0, columns8 ? (use_tile_parallel3 ? 8 : 7) : (dense_bitcast_route ? (block32_mode == 1 ? 3 : 1 + dense_tile4) : 0));
}

/**
 * @brief Applies a quantized matrix multiplication using N-bit precision.
 *
 * This function computes the matrix multiplication of the quantized tensor inputs with multiple
 * optional optimizations tailored to the GPU backend. Depending on the provided parameters and GPU
 * capabilities, it selects one of several optimized kernels (such as subgroup matrix multiplication,
 * DP4A, wide tile programs, or the default matmul program) to perform the computation.
 * It can be called by the MatMulNBits operator or directly for custom scenarios like QMoe.
 *
 * @param a              Pointer to the left-hand side (activation) tensor.
 * @param b              Pointer to the quantized weight tensor.
 *                       b has the shape (N, k_blocks, blob_size) or (weight_batch, N, k_blocks, blob_size)
 * @param scales         Pointer to the tensor containing scaling factors for quantization.
 *                       scales has the shape (N) or (weight_batch, N)
 * @param zero_points    Pointer to the zero-point tensor for quantization; must be of type uint8 if provided.
 *                       weight_index > 0 is only supported when zero_points is nullptr.
 * @param bias           Pointer to the bias tensor; optional.
 * @param K_op           The K dimension of the operation (number of columns in 'a' and rows in 'b' before quantization).
 * @param N_op           The N dimension of the operation (number of columns in 'b').
 * @param block_size_op  The block size used for quantization partitioning.
 * @param accuracy_level Accuracy level influencing the choice of optimized kernel.
 * @param nbits          Number of bits used for quantization.
 * @param weight_index   Index of the weight matrix in case of stacked weights; defaults to 0.
 * @param context        Compute context for WebGPU, providing device-specific information and execution facilities.
 * @param y              Pointer to the output tensor that will hold the result.
 *
 * @return Status indicating whether the operation was successful or if an error occurred.
 *
 * @note Special optimizations are considered:
 *       - Subgroup matrix multiplication for GPUs with supported configs.
 *       - DP4A-based multiplication on FP32-only GPUs for specific dimensions and conditions.
 *       - A wide tile program is used when block size, component count, and other criteria are met.
 *       - Otherwise, a default matmul program is used.
 */
Status ApplyMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points, const Tensor* bias,
                        int64_t K_op,
                        int64_t N_op,
                        int64_t block_size_op,
                        int64_t accuracy_level,
                        int64_t nbits,
                        onnxruntime::webgpu::ComputeContext& context,
                        Tensor* y,
                        const uint32_t weight_index,
                        const Tensor* weight_index_indirect,
                        uint32_t override_M) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  const bool has_bias = bias != nullptr;
  const bool has_weight_idx_indirect = weight_index_indirect != nullptr;
  const bool has_weight_idx = weight_index > 0 || has_weight_idx_indirect;
  const bool has_zero_points = zero_points != nullptr;
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(), "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
  }

  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t dispatch_M = (override_M > 0) ? override_M : M;
  const bool broadcast_a = dispatch_M > M;
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_op);


  const int grouped_prefill_rows = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_GROUPED_PREFILL", 0);
  ORT_RETURN_IF_NOT(grouped_prefill_rows == 0 || grouped_prefill_rows == 1,
                    "ORT_WEBGPU_QMOE_GROUPED_PREFILL must be 0 or 1");
  if (grouped_prefill_rows == 1 && batch_count == 1 && dispatch_M == M &&
      (M == 5120 || M == 10240) && ((K == 2560 && N == 1280) || (K == 640 && N == 2560)) &&
      nbits == 4 && block_size == 32 && has_weight_idx_indirect && !has_zero_points &&
      a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
      (bias == nullptr || bias->IsDataType<MLFloat16>()) &&
      !context.EnableMatmulFp32Accumulation() &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"}) {
    GroupedPrefillRows4Program program{has_bias};
    program.SetWorkgroupSize(128).SetDispatchGroupSize(N / 8, M / 4);
    program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 4},
                        {b, ProgramTensorMetadataDependency::TypeAndRank, 16},
                        {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}});
    if (has_bias) program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank, 1});
    program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, 1})
        .AddUniformVariables({{N}, {K}, {N / 8}})
        .CacheHint(has_bias);
    return context.RunProgram(program);
  }

  if (nbits == 3) {
    ORT_RETURN_IF_NOT(batch_count == 1 && block_size == 32 && K % 32 == 0 && !has_zero_points,
                      "WebGPU INT3 MatMulNBits requires one batch, block32, K divisible by32 and implicit zero points.");
    ORT_RETURN_IF_NOT(a->DataType() == scales->DataType(),
                      "WebGPU INT3 activations and scales must have the same floating-point type.");
    const auto max_groups = context.DeviceLimits().maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(N <= max_groups && dispatch_M <= max_groups,
                      "WebGPU INT3 dispatch exceeds per-dimension workgroup limits.");
    MatMulInt3Program program{has_weight_idx_indirect, has_bias};
    program.SetWorkgroupSize(128).SetDispatchGroupSize(N, dispatch_M);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_weight_idx_indirect) program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_bias) program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank});
    program.AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
        .AddUniformVariables({N, K, weight_index, static_cast<uint32_t>(broadcast_a)})
        .CacheHint(has_weight_idx_indirect, has_bias);
    return context.RunProgram(program);
  }

  if (ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXACT_DENSE_MID", 0) != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M == 1 && dispatch_M == M && K == 2560 && (N == 512 || N == 640) &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      !context.EnableMatmulFp32Accumulation() &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    ExactDenseMidProgram program;
    program.SetWorkgroupSize(64).SetDispatchGroupSize(N / 2);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
    return context.RunProgram(program);
  }

  if (ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_EXACT_DENSE", 0) != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M == 1 && dispatch_M == M && K == 10240 && N == 324 &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      !context.EnableMatmulFp32Accumulation() &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    HCExactDenseProgram program;
    program.SetWorkgroupSize(64).SetDispatchGroupSize(162);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
    return context.RunProgram(program);
  }

  // Adapt the few-row, K-parallel GEMV strategy from Tianlei's skinny GEMM to
  // symmetric RTN INT4. Keep unsupported layouts and precisions on existing paths.
  if ((ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_SKINNY", 0) != 0 ||
       (K == 10240 && N == 324 &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY", 0) != 0)) &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M <= 4 && dispatch_M == M && K >= 2048 && K % 32 == 0 && N <= 1024 &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    MatMulNBitsSkinnyProgram program;
    program.SetWorkgroupSize(128).SetDispatchGroupSize(N * M);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
        .AddUniformVariables({{M}, {N}, {K}});
    return context.RunProgram(program);
  }

  // Special case matrix used by bitnets where there is a single scale for the entire
  const bool single_scale_weights = (block_size == K * N);
  const uint32_t block_size_per_col = single_scale_weights ? K : block_size;
  const uint32_t n_blocks_per_col = (K + block_size_per_col - 1) / block_size_per_col;
  const uint32_t blob_size = (block_size_per_col / 8) * static_cast<uint32_t>(nbits);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);
  uint32_t components = GetMaxComponents(N);
  // zero_points has shape[N * CeilDiv(n_blocks_per_col * bits, 8)].
  // The shader uses a flat linear index to address individual n-bit zero point values.
  // Since each column's zero points are byte-aligned in the packed buffer, we must round
  // n_blocks_per_col up to the next multiple of (8/nbits) — the number of zero point
  // values per byte — so that the linear stride correctly skips byte-boundary padding.
  const uint32_t zp_elements_per_byte = 8 / static_cast<uint32_t>(nbits);
  uint32_t zero_blocks_per_col = (n_blocks_per_col + zp_elements_per_byte - 1) / zp_elements_per_byte * zp_elements_per_byte;

  // apple|intel - Experimental dawn support for subgroup matrix matmul.
  std::optional<SubgroupMatrixConfig> subgroup_matrix_config;
  if (CanApplySubgroupMatrixMatMulNBits(context,
                                        accuracy_level,
                                        block_size,
                                        batch_count,
                                        N,
                                        K,
                                        static_cast<uint32_t>(nbits),
                                        y->DataType() == DataTypeImpl::GetType<MLFloat16>(),
                                        subgroup_matrix_config,
                                        M,
                                        has_weight_idx_indirect)) {
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, M, N, K,
                                          static_cast<uint32_t>(nbits), zero_blocks_per_col,
                                          *subgroup_matrix_config, context, y, weight_index,
                                          weight_index_indirect);
  }

  // On FP32 only GPUs and Qualcomm GPUs, integer math is faster than FP32 therefore always use DP4A independent of length of M.
  // DP4A Q2 path now supports custom zero points via a 1024-entry LUT (4 zero-point sections × 256 byte values).
  if (CanApplyDP4AMatrixMatMulNBits(context, accuracy_level, block_size, N, K, components_a,
                                    M, has_weight_idx_indirect, y)) {
    return ApplyDP4AMatrixMatMulNBits(a, b, scales, zero_points, bias, batch_count, M, dispatch_M, N, K, block_size, zero_blocks_per_col, kMinMForTileOptimization, static_cast<uint32_t>(nbits), context, y, weight_index, weight_index_indirect);
  }

  // WideTileProgram
  // This program is optimized for Block32 prefill.
  const bool use_wide_tile_program = CanApplyWideTileMatMulNBits(M,
                                                                 K,
                                                                 block_size,
                                                                 nbits,
                                                                 has_weight_idx_indirect,
                                                                 components_a,
                                                                 components_b);

  const bool acc_f32 = context.EnableMatmulFp32Accumulation();

  if (use_wide_tile_program) {
    // Enforce output components to 1.
    components = 1;

    constexpr uint32_t workgroup_size = 128;
    constexpr uint32_t tile_n = workgroup_size;
    const bool is_f16 = a->DataType() == DataTypeImpl::GetType<MLFloat16>();
    const uint32_t tile_m = is_f16 ? 2 * workgroup_size / 8 : workgroup_size / 8;
    const uint32_t num_N_tile = CeilDiv(N, tile_n);
    const uint32_t num_M_tile = CeilDiv(dispatch_M, tile_m);

    bool is_nvidia = context.AdapterInfo().vendor == std::string_view{"nvidia"};

    // The template shuffles tile_m lanes in bands of subgroup_min_size when that's a
    // width it supports; otherwise (including when Subgroups isn't supported) it falls
    // back to a direct reduction.
    // NOTE: NVIDIA GPUs prefer direct reduction.
    const uint32_t subgroup_min_size = (context.HasFeature(wgpu::FeatureName::Subgroups) && !is_nvidia)
                                           ? context.AdapterInfo().subgroupMinSize
                                           : 0u;

    MatMulNBitsWideTileProgram program{has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, tile_m, tile_n, static_cast<uint32_t>(nbits), subgroup_min_size, acc_f32};
    program.SetWorkgroupSize(workgroup_size);
    program.SetDispatchGroupSize(num_N_tile, num_M_tile, batch_count);

    constexpr uint32_t kU32Components = 4;
    const uint32_t components_b_with_u32 = components_b * kU32Components;
    const uint32_t K_of_b = n_blocks_per_col * blob_size / components_b_with_u32;
    const uint32_t K_of_a = K / components_a;

    program.AddInput({a,
                      ProgramTensorMetadataDependency::TypeAndRank,
                      onnxruntime::narrow<int>(components_a)});
    program.AddInput({b,
                      ProgramTensorMetadataDependency::TypeAndRank,
                      onnxruntime::narrow<int>(components_b_with_u32)});
    program.AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_zero_points) {
      program.AddInput({zero_points,
                        ProgramTensorMetadataDependency::TypeAndRank,
                        {CeilDiv(zero_points->Shape().Size(), static_cast<int64_t>(4))},
                        4});
    }
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    if (has_weight_idx_indirect) {
      program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
    }
    program.AddOutput({y,
                       ProgramTensorMetadataDependency::TypeAndRank,
                       onnxruntime::narrow<int>(components)});
    program.AddUniformVariables({{batch_count},
                                 {dispatch_M},
                                 {N},
                                 {K_of_a},
                                 {K_of_b},
                                 {n_blocks_per_col},
                                 {zero_blocks_per_col},
                                 {num_N_tile},
                                 {num_M_tile},
                                 {weight_index}});
    program.CacheHint(nbits, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, subgroup_min_size, acc_f32);

    return context.RunProgram(program);
  }

  // Use tile_size_k_vec=32 by default for better K-dimension parallelism.
  // Intel devices use 16 as they have different subgroup/cache characteristics.
  const uint32_t tile_size_k_vec =
      (context.AdapterInfo().vendor == std::string_view{"intel"}) ? 16u : 32u;

  constexpr uint32_t workgroup_size = 128;
  constexpr uint32_t tile_size = 8;
  constexpr uint32_t kU32Components = 4;
  uint32_t components_b_with_u32 = components_b * kU32Components;
  uint32_t K_of_b = (n_blocks_per_col * blob_size) / components_b_with_u32;
  MatMulNBitsProgram program{tile_size, static_cast<uint32_t>(nbits), has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, single_scale_weights, tile_size_k_vec, broadcast_a, acc_f32};
  program.SetWorkgroupSize(workgroup_size);
  uint32_t num_N_tile = (N + tile_size - 1) / tile_size;
  program.SetDispatchGroupSize(num_N_tile, dispatch_M, batch_count);
  program
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                  {scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariables({{M},
                            {N},
                            {K},
                            {K / components_a},
                            {K_of_b},
                            {block_size},
                            {n_blocks_per_col},
                            {zero_blocks_per_col},
                            {num_N_tile},
                            {batch_count},
                            {weight_index},
                            {dispatch_M}})
      .CacheHint(nbits, has_zero_points, single_scale_weights, has_bias, has_weight_idx, has_weight_idx_indirect, tile_size_k_vec, broadcast_a, acc_f32);
  if (has_zero_points) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_weight_idx_indirect) {
    program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
  }
  return context.RunProgram(program);
}

namespace {
Status ApplyDenseBitcastMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points, const Tensor* bias,
                        int64_t K_op,
                        int64_t N_op,
                        int64_t block_size_op,
                        int64_t accuracy_level,
                        int64_t nbits,
                        onnxruntime::webgpu::ComputeContext& context,
                        Tensor* y,
                        const uint32_t weight_index,
                        const Tensor* weight_index_indirect,
                        uint32_t override_M) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  const bool has_bias = bias != nullptr;
  const bool has_weight_idx_indirect = weight_index_indirect != nullptr;
  const bool has_weight_idx = weight_index > 0 || has_weight_idx_indirect;
  const bool has_zero_points = zero_points != nullptr;
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(), "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
  }

  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t dispatch_M = (override_M > 0) ? override_M : M;
  const bool broadcast_a = dispatch_M > M;
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_op);

  if (nbits == 3) {
    ORT_RETURN_IF_NOT(batch_count == 1 && block_size == 32 && K % 32 == 0 && !has_zero_points,
                      "WebGPU INT3 MatMulNBits requires one batch, block32, K divisible by32 and implicit zero points.");
    ORT_RETURN_IF_NOT(a->DataType() == scales->DataType(),
                      "WebGPU INT3 activations and scales must have the same floating-point type.");
    const auto max_groups = context.DeviceLimits().maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(N <= max_groups && dispatch_M <= max_groups,
                      "WebGPU INT3 dispatch exceeds per-dimension workgroup limits.");
    MatMulInt3Program program{has_weight_idx_indirect, has_bias};
    program.SetWorkgroupSize(128).SetDispatchGroupSize(N, dispatch_M);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_weight_idx_indirect) program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_bias) program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank});
    program.AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
        .AddUniformVariables({N, K, weight_index, static_cast<uint32_t>(broadcast_a)})
        .CacheHint(has_weight_idx_indirect, has_bias);
    return context.RunProgram(program);
  }

  if (ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXACT_DENSE_MID", 0) != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M == 1 && dispatch_M == M && K == 2560 && (N == 512 || N == 640) &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      !context.EnableMatmulFp32Accumulation() &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    ExactDenseMidProgram program;
    program.SetWorkgroupSize(64).SetDispatchGroupSize(N / 2);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
    return context.RunProgram(program);
  }

  if (ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_EXACT_DENSE", 0) != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M == 1 && dispatch_M == M && K == 10240 && N == 324 &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      !context.EnableMatmulFp32Accumulation() &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    HCExactDenseProgram program;
    program.SetWorkgroupSize(64).SetDispatchGroupSize(162);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
    return context.RunProgram(program);
  }

  // Adapt the few-row, K-parallel GEMV strategy from Tianlei's skinny GEMM to
  // symmetric RTN INT4. Keep unsupported layouts and precisions on existing paths.
  if ((ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_SKINNY", 0) != 0 ||
       (K == 10240 && N == 324 &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY", 0) != 0)) &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M <= 4 && dispatch_M == M && K >= 2048 && K % 32 == 0 && N <= 1024 &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    MatMulNBitsSkinnyProgram program;
    program.SetWorkgroupSize(128).SetDispatchGroupSize(N * M);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
        .AddUniformVariables({{M}, {N}, {K}});
    return context.RunProgram(program);
  }

  // Special case matrix used by bitnets where there is a single scale for the entire
  const bool single_scale_weights = (block_size == K * N);
  const uint32_t block_size_per_col = single_scale_weights ? K : block_size;
  const uint32_t n_blocks_per_col = (K + block_size_per_col - 1) / block_size_per_col;
  const uint32_t blob_size = (block_size_per_col / 8) * static_cast<uint32_t>(nbits);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);
  uint32_t components = GetMaxComponents(N);
  // zero_points has shape[N * CeilDiv(n_blocks_per_col * bits, 8)].
  // The shader uses a flat linear index to address individual n-bit zero point values.
  // Since each column's zero points are byte-aligned in the packed buffer, we must round
  // n_blocks_per_col up to the next multiple of (8/nbits) — the number of zero point
  // values per byte — so that the linear stride correctly skips byte-boundary padding.
  const uint32_t zp_elements_per_byte = 8 / static_cast<uint32_t>(nbits);
  uint32_t zero_blocks_per_col = (n_blocks_per_col + zp_elements_per_byte - 1) / zp_elements_per_byte * zp_elements_per_byte;

  // apple|intel - Experimental dawn support for subgroup matrix matmul.
  std::optional<SubgroupMatrixConfig> subgroup_matrix_config;
  if (CanApplySubgroupMatrixMatMulNBits(context,
                                        accuracy_level,
                                        block_size,
                                        batch_count,
                                        N,
                                        K,
                                        static_cast<uint32_t>(nbits),
                                        y->DataType() == DataTypeImpl::GetType<MLFloat16>(),
                                        subgroup_matrix_config,
                                        M,
                                        has_weight_idx_indirect)) {
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, M, N, K,
                                          static_cast<uint32_t>(nbits), zero_blocks_per_col,
                                          *subgroup_matrix_config, context, y, weight_index,
                                          weight_index_indirect);
  }

  // On FP32 only GPUs and Qualcomm GPUs, integer math is faster than FP32 therefore always use DP4A independent of length of M.
  // DP4A Q2 path now supports custom zero points via a 1024-entry LUT (4 zero-point sections × 256 byte values).
  if (CanApplyDP4AMatrixMatMulNBits(context, accuracy_level, block_size, N, K, components_a,
                                    M, has_weight_idx_indirect, y)) {
    return ApplyDP4AMatrixMatMulNBits(a, b, scales, zero_points, bias, batch_count, M, dispatch_M, N, K, block_size, zero_blocks_per_col, kMinMForTileOptimization, static_cast<uint32_t>(nbits), context, y, weight_index, weight_index_indirect);
  }

  // WideTileProgram
  // This program is optimized for Block32 prefill.
  const bool use_wide_tile_program = CanApplyWideTileMatMulNBits(M,
                                                                 K,
                                                                 block_size,
                                                                 nbits,
                                                                 has_weight_idx_indirect,
                                                                 components_a,
                                                                 components_b);

  const bool acc_f32 = context.EnableMatmulFp32Accumulation();

  if (use_wide_tile_program) {
    // Enforce output components to 1.
    components = 1;

    constexpr uint32_t workgroup_size = 128;
    constexpr uint32_t tile_n = workgroup_size;
    const bool is_f16 = a->DataType() == DataTypeImpl::GetType<MLFloat16>();
    const uint32_t tile_m = is_f16 ? 2 * workgroup_size / 8 : workgroup_size / 8;
    const uint32_t num_N_tile = CeilDiv(N, tile_n);
    const uint32_t num_M_tile = CeilDiv(dispatch_M, tile_m);

    bool is_nvidia = context.AdapterInfo().vendor == std::string_view{"nvidia"};

    // The template shuffles tile_m lanes in bands of subgroup_min_size when that's a
    // width it supports; otherwise (including when Subgroups isn't supported) it falls
    // back to a direct reduction.
    // NOTE: NVIDIA GPUs prefer direct reduction.
    const uint32_t subgroup_min_size = (context.HasFeature(wgpu::FeatureName::Subgroups) && !is_nvidia)
                                           ? context.AdapterInfo().subgroupMinSize
                                           : 0u;

    MatMulNBitsWideTileProgram program{has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, tile_m, tile_n, static_cast<uint32_t>(nbits), subgroup_min_size, acc_f32};
    program.SetWorkgroupSize(workgroup_size);
    program.SetDispatchGroupSize(num_N_tile, num_M_tile, batch_count);

    constexpr uint32_t kU32Components = 4;
    const uint32_t components_b_with_u32 = components_b * kU32Components;
    const uint32_t K_of_b = n_blocks_per_col * blob_size / components_b_with_u32;
    const uint32_t K_of_a = K / components_a;

    program.AddInput({a,
                      ProgramTensorMetadataDependency::TypeAndRank,
                      onnxruntime::narrow<int>(components_a)});
    program.AddInput({b,
                      ProgramTensorMetadataDependency::TypeAndRank,
                      onnxruntime::narrow<int>(components_b_with_u32)});
    program.AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_zero_points) {
      program.AddInput({zero_points,
                        ProgramTensorMetadataDependency::TypeAndRank,
                        {CeilDiv(zero_points->Shape().Size(), static_cast<int64_t>(4))},
                        4});
    }
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    if (has_weight_idx_indirect) {
      program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
    }
    program.AddOutput({y,
                       ProgramTensorMetadataDependency::TypeAndRank,
                       onnxruntime::narrow<int>(components)});
    program.AddUniformVariables({{batch_count},
                                 {dispatch_M},
                                 {N},
                                 {K_of_a},
                                 {K_of_b},
                                 {n_blocks_per_col},
                                 {zero_blocks_per_col},
                                 {num_N_tile},
                                 {num_M_tile},
                                 {weight_index}});
    program.CacheHint(nbits, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, subgroup_min_size, acc_f32);

    return context.RunProgram(program);
  }

  // Use tile_size_k_vec=32 by default for better K-dimension parallelism.
  // Intel devices use 16 as they have different subgroup/cache characteristics.
  const uint32_t tile_size_k_vec =
      (context.AdapterInfo().vendor == std::string_view{"intel"}) ? 16u : 32u;

  constexpr uint32_t workgroup_size = 128;
  constexpr uint32_t tile_size = 8;
  constexpr uint32_t kU32Components = 4;
  uint32_t components_b_with_u32 = components_b * kU32Components;
  uint32_t K_of_b = (n_blocks_per_col * blob_size) / components_b_with_u32;
  DenseBitcastProgram program{tile_size, static_cast<uint32_t>(nbits), has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, single_scale_weights, tile_size_k_vec, broadcast_a, acc_f32};
  program.SetWorkgroupSize(workgroup_size);
  uint32_t num_N_tile = (N + tile_size - 1) / tile_size;
  program.SetDispatchGroupSize(num_N_tile, dispatch_M, batch_count);
  program
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                  {scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariables({{M},
                            {N},
                            {K},
                            {K / components_a},
                            {K_of_b},
                            {block_size},
                            {n_blocks_per_col},
                            {zero_blocks_per_col},
                            {num_N_tile},
                            {batch_count},
                            {weight_index},
                            {dispatch_M}})
      .CacheHint(nbits, has_zero_points, single_scale_weights, has_bias, has_weight_idx, has_weight_idx_indirect, tile_size_k_vec, broadcast_a, acc_f32);
  if (has_zero_points) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_weight_idx_indirect) {
    program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
  }
  return context.RunProgram(program);
}


Status ApplyDenseTile4MatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points, const Tensor* bias,
                        int64_t K_op,
                        int64_t N_op,
                        int64_t block_size_op,
                        int64_t accuracy_level,
                        int64_t nbits,
                        onnxruntime::webgpu::ComputeContext& context,
                        Tensor* y,
                        const uint32_t weight_index,
                        const Tensor* weight_index_indirect,
                        uint32_t override_M) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  const bool has_bias = bias != nullptr;
  const bool has_weight_idx_indirect = weight_index_indirect != nullptr;
  const bool has_weight_idx = weight_index > 0 || has_weight_idx_indirect;
  const bool has_zero_points = zero_points != nullptr;
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(), "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
  }

  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t dispatch_M = (override_M > 0) ? override_M : M;
  const bool broadcast_a = dispatch_M > M;
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_op);

  if (nbits == 3) {
    ORT_RETURN_IF_NOT(batch_count == 1 && block_size == 32 && K % 32 == 0 && !has_zero_points,
                      "WebGPU INT3 MatMulNBits requires one batch, block32, K divisible by32 and implicit zero points.");
    ORT_RETURN_IF_NOT(a->DataType() == scales->DataType(),
                      "WebGPU INT3 activations and scales must have the same floating-point type.");
    const auto max_groups = context.DeviceLimits().maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(N <= max_groups && dispatch_M <= max_groups,
                      "WebGPU INT3 dispatch exceeds per-dimension workgroup limits.");
    MatMulInt3Program program{has_weight_idx_indirect, has_bias};
    program.SetWorkgroupSize(128).SetDispatchGroupSize(N, dispatch_M);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_weight_idx_indirect) program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_bias) program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank});
    program.AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
        .AddUniformVariables({N, K, weight_index, static_cast<uint32_t>(broadcast_a)})
        .CacheHint(has_weight_idx_indirect, has_bias);
    return context.RunProgram(program);
  }

  if (ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXACT_DENSE_MID", 0) != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M == 1 && dispatch_M == M && K == 2560 && (N == 512 || N == 640) &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      !context.EnableMatmulFp32Accumulation() &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    ExactDenseMidProgram program;
    program.SetWorkgroupSize(64).SetDispatchGroupSize(N / 2);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
    return context.RunProgram(program);
  }

  if (ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_EXACT_DENSE", 0) != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M == 1 && dispatch_M == M && K == 10240 && N == 324 &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      !context.EnableMatmulFp32Accumulation() &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    HCExactDenseProgram program;
    program.SetWorkgroupSize(64).SetDispatchGroupSize(162);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
    return context.RunProgram(program);
  }

  // Adapt the few-row, K-parallel GEMV strategy from Tianlei's skinny GEMM to
  // symmetric RTN INT4. Keep unsupported layouts and precisions on existing paths.
  if ((ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_SKINNY", 0) != 0 ||
       (K == 10240 && N == 324 &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY", 0) != 0)) &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M <= 4 && dispatch_M == M && K >= 2048 && K % 32 == 0 && N <= 1024 &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    MatMulNBitsSkinnyProgram program;
    program.SetWorkgroupSize(128).SetDispatchGroupSize(N * M);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
        .AddUniformVariables({{M}, {N}, {K}});
    return context.RunProgram(program);
  }

  // Special case matrix used by bitnets where there is a single scale for the entire
  const bool single_scale_weights = (block_size == K * N);
  const uint32_t block_size_per_col = single_scale_weights ? K : block_size;
  const uint32_t n_blocks_per_col = (K + block_size_per_col - 1) / block_size_per_col;
  const uint32_t blob_size = (block_size_per_col / 8) * static_cast<uint32_t>(nbits);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);
  uint32_t components = GetMaxComponents(N);
  // zero_points has shape[N * CeilDiv(n_blocks_per_col * bits, 8)].
  // The shader uses a flat linear index to address individual n-bit zero point values.
  // Since each column's zero points are byte-aligned in the packed buffer, we must round
  // n_blocks_per_col up to the next multiple of (8/nbits) — the number of zero point
  // values per byte — so that the linear stride correctly skips byte-boundary padding.
  const uint32_t zp_elements_per_byte = 8 / static_cast<uint32_t>(nbits);
  uint32_t zero_blocks_per_col = (n_blocks_per_col + zp_elements_per_byte - 1) / zp_elements_per_byte * zp_elements_per_byte;

  // apple|intel - Experimental dawn support for subgroup matrix matmul.
  std::optional<SubgroupMatrixConfig> subgroup_matrix_config;
  if (CanApplySubgroupMatrixMatMulNBits(context,
                                        accuracy_level,
                                        block_size,
                                        batch_count,
                                        N,
                                        K,
                                        static_cast<uint32_t>(nbits),
                                        y->DataType() == DataTypeImpl::GetType<MLFloat16>(),
                                        subgroup_matrix_config,
                                        M,
                                        has_weight_idx_indirect)) {
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, M, N, K,
                                          static_cast<uint32_t>(nbits), zero_blocks_per_col,
                                          *subgroup_matrix_config, context, y, weight_index,
                                          weight_index_indirect);
  }

  // On FP32 only GPUs and Qualcomm GPUs, integer math is faster than FP32 therefore always use DP4A independent of length of M.
  // DP4A Q2 path now supports custom zero points via a 1024-entry LUT (4 zero-point sections × 256 byte values).
  if (CanApplyDP4AMatrixMatMulNBits(context, accuracy_level, block_size, N, K, components_a,
                                    M, has_weight_idx_indirect, y)) {
    return ApplyDP4AMatrixMatMulNBits(a, b, scales, zero_points, bias, batch_count, M, dispatch_M, N, K, block_size, zero_blocks_per_col, kMinMForTileOptimization, static_cast<uint32_t>(nbits), context, y, weight_index, weight_index_indirect);
  }

  // WideTileProgram
  // This program is optimized for Block32 prefill.
  const bool use_wide_tile_program = CanApplyWideTileMatMulNBits(M,
                                                                 K,
                                                                 block_size,
                                                                 nbits,
                                                                 has_weight_idx_indirect,
                                                                 components_a,
                                                                 components_b);

  const bool acc_f32 = context.EnableMatmulFp32Accumulation();

  if (use_wide_tile_program) {
    // Enforce output components to 1.
    components = 1;

    constexpr uint32_t workgroup_size = 128;
    constexpr uint32_t tile_n = workgroup_size;
    const bool is_f16 = a->DataType() == DataTypeImpl::GetType<MLFloat16>();
    const uint32_t tile_m = is_f16 ? 2 * workgroup_size / 8 : workgroup_size / 8;
    const uint32_t num_N_tile = CeilDiv(N, tile_n);
    const uint32_t num_M_tile = CeilDiv(dispatch_M, tile_m);

    bool is_nvidia = context.AdapterInfo().vendor == std::string_view{"nvidia"};

    // The template shuffles tile_m lanes in bands of subgroup_min_size when that's a
    // width it supports; otherwise (including when Subgroups isn't supported) it falls
    // back to a direct reduction.
    // NOTE: NVIDIA GPUs prefer direct reduction.
    const uint32_t subgroup_min_size = (context.HasFeature(wgpu::FeatureName::Subgroups) && !is_nvidia)
                                           ? context.AdapterInfo().subgroupMinSize
                                           : 0u;

    MatMulNBitsWideTileProgram program{has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, tile_m, tile_n, static_cast<uint32_t>(nbits), subgroup_min_size, acc_f32};
    program.SetWorkgroupSize(workgroup_size);
    program.SetDispatchGroupSize(num_N_tile, num_M_tile, batch_count);

    constexpr uint32_t kU32Components = 4;
    const uint32_t components_b_with_u32 = components_b * kU32Components;
    const uint32_t K_of_b = n_blocks_per_col * blob_size / components_b_with_u32;
    const uint32_t K_of_a = K / components_a;

    program.AddInput({a,
                      ProgramTensorMetadataDependency::TypeAndRank,
                      onnxruntime::narrow<int>(components_a)});
    program.AddInput({b,
                      ProgramTensorMetadataDependency::TypeAndRank,
                      onnxruntime::narrow<int>(components_b_with_u32)});
    program.AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_zero_points) {
      program.AddInput({zero_points,
                        ProgramTensorMetadataDependency::TypeAndRank,
                        {CeilDiv(zero_points->Shape().Size(), static_cast<int64_t>(4))},
                        4});
    }
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    if (has_weight_idx_indirect) {
      program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
    }
    program.AddOutput({y,
                       ProgramTensorMetadataDependency::TypeAndRank,
                       onnxruntime::narrow<int>(components)});
    program.AddUniformVariables({{batch_count},
                                 {dispatch_M},
                                 {N},
                                 {K_of_a},
                                 {K_of_b},
                                 {n_blocks_per_col},
                                 {zero_blocks_per_col},
                                 {num_N_tile},
                                 {num_M_tile},
                                 {weight_index}});
    program.CacheHint(nbits, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, subgroup_min_size, acc_f32);

    return context.RunProgram(program);
  }

  // Use tile_size_k_vec=32 by default for better K-dimension parallelism.
  // Intel devices use 16 as they have different subgroup/cache characteristics.
  const uint32_t tile_size_k_vec =
      (context.AdapterInfo().vendor == std::string_view{"intel"}) ? 16u : 32u;

  constexpr uint32_t workgroup_size = 128;
  constexpr uint32_t tile_size = 4;
  constexpr uint32_t kU32Components = 4;
  uint32_t components_b_with_u32 = components_b * kU32Components;
  uint32_t K_of_b = (n_blocks_per_col * blob_size) / components_b_with_u32;
  DenseTile4Program program{tile_size, static_cast<uint32_t>(nbits), has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, single_scale_weights, tile_size_k_vec, broadcast_a, acc_f32};
  program.SetWorkgroupSize(workgroup_size);
  uint32_t num_N_tile = (N + tile_size - 1) / tile_size;
  program.SetDispatchGroupSize(num_N_tile, dispatch_M, batch_count);
  program
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                  {scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariables({{M},
                            {N},
                            {K},
                            {K / components_a},
                            {K_of_b},
                            {block_size},
                            {n_blocks_per_col},
                            {zero_blocks_per_col},
                            {num_N_tile},
                            {batch_count},
                            {weight_index},
                            {dispatch_M}})
      .CacheHint(nbits, has_zero_points, single_scale_weights, has_bias, has_weight_idx, has_weight_idx_indirect, tile_size_k_vec, broadcast_a, acc_f32);
  if (has_zero_points) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_weight_idx_indirect) {
    program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
  }
  return context.RunProgram(program);
}


Status ApplyDenseBlock32MatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points, const Tensor* bias,
                        int64_t K_op,
                        int64_t N_op,
                        int64_t block_size_op,
                        int64_t accuracy_level,
                        int64_t nbits,
                        onnxruntime::webgpu::ComputeContext& context,
                        Tensor* y,
                        const uint32_t weight_index,
                        const Tensor* weight_index_indirect,
                        uint32_t override_M) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  const bool has_bias = bias != nullptr;
  const bool has_weight_idx_indirect = weight_index_indirect != nullptr;
  const bool has_weight_idx = weight_index > 0 || has_weight_idx_indirect;
  const bool has_zero_points = zero_points != nullptr;
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(), "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
  }

  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t dispatch_M = (override_M > 0) ? override_M : M;
  const bool broadcast_a = dispatch_M > M;
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_op);

  if (nbits == 3) {
    ORT_RETURN_IF_NOT(batch_count == 1 && block_size == 32 && K % 32 == 0 && !has_zero_points,
                      "WebGPU INT3 MatMulNBits requires one batch, block32, K divisible by32 and implicit zero points.");
    ORT_RETURN_IF_NOT(a->DataType() == scales->DataType(),
                      "WebGPU INT3 activations and scales must have the same floating-point type.");
    const auto max_groups = context.DeviceLimits().maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(N <= max_groups && dispatch_M <= max_groups,
                      "WebGPU INT3 dispatch exceeds per-dimension workgroup limits.");
    MatMulInt3Program program{has_weight_idx_indirect, has_bias};
    program.SetWorkgroupSize(128).SetDispatchGroupSize(N, dispatch_M);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_weight_idx_indirect) program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_bias) program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank});
    program.AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
        .AddUniformVariables({N, K, weight_index, static_cast<uint32_t>(broadcast_a)})
        .CacheHint(has_weight_idx_indirect, has_bias);
    return context.RunProgram(program);
  }

  if (ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXACT_DENSE_MID", 0) != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M == 1 && dispatch_M == M && K == 2560 && (N == 512 || N == 640) &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      !context.EnableMatmulFp32Accumulation() &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    ExactDenseMidProgram program;
    program.SetWorkgroupSize(64).SetDispatchGroupSize(N / 2);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
    return context.RunProgram(program);
  }

  if (ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_EXACT_DENSE", 0) != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M == 1 && dispatch_M == M && K == 10240 && N == 324 &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      !context.EnableMatmulFp32Accumulation() &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    HCExactDenseProgram program;
    program.SetWorkgroupSize(64).SetDispatchGroupSize(162);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
    return context.RunProgram(program);
  }

  // Adapt the few-row, K-parallel GEMV strategy from Tianlei's skinny GEMM to
  // symmetric RTN INT4. Keep unsupported layouts and precisions on existing paths.
  if ((ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_SKINNY", 0) != 0 ||
       (K == 10240 && N == 324 &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY", 0) != 0)) &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M <= 4 && dispatch_M == M && K >= 2048 && K % 32 == 0 && N <= 1024 &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    MatMulNBitsSkinnyProgram program;
    program.SetWorkgroupSize(128).SetDispatchGroupSize(N * M);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
        .AddUniformVariables({{M}, {N}, {K}});
    return context.RunProgram(program);
  }

  // Special case matrix used by bitnets where there is a single scale for the entire
  const bool single_scale_weights = (block_size == K * N);
  const uint32_t block_size_per_col = single_scale_weights ? K : block_size;
  const uint32_t n_blocks_per_col = (K + block_size_per_col - 1) / block_size_per_col;
  const uint32_t blob_size = (block_size_per_col / 8) * static_cast<uint32_t>(nbits);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);
  uint32_t components = GetMaxComponents(N);
  // zero_points has shape[N * CeilDiv(n_blocks_per_col * bits, 8)].
  // The shader uses a flat linear index to address individual n-bit zero point values.
  // Since each column's zero points are byte-aligned in the packed buffer, we must round
  // n_blocks_per_col up to the next multiple of (8/nbits) — the number of zero point
  // values per byte — so that the linear stride correctly skips byte-boundary padding.
  const uint32_t zp_elements_per_byte = 8 / static_cast<uint32_t>(nbits);
  uint32_t zero_blocks_per_col = (n_blocks_per_col + zp_elements_per_byte - 1) / zp_elements_per_byte * zp_elements_per_byte;

  // apple|intel - Experimental dawn support for subgroup matrix matmul.
  std::optional<SubgroupMatrixConfig> subgroup_matrix_config;
  if (CanApplySubgroupMatrixMatMulNBits(context,
                                        accuracy_level,
                                        block_size,
                                        batch_count,
                                        N,
                                        K,
                                        static_cast<uint32_t>(nbits),
                                        y->DataType() == DataTypeImpl::GetType<MLFloat16>(),
                                        subgroup_matrix_config,
                                        M,
                                        has_weight_idx_indirect)) {
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, M, N, K,
                                          static_cast<uint32_t>(nbits), zero_blocks_per_col,
                                          *subgroup_matrix_config, context, y, weight_index,
                                          weight_index_indirect);
  }

  // On FP32 only GPUs and Qualcomm GPUs, integer math is faster than FP32 therefore always use DP4A independent of length of M.
  // DP4A Q2 path now supports custom zero points via a 1024-entry LUT (4 zero-point sections × 256 byte values).
  if (CanApplyDP4AMatrixMatMulNBits(context, accuracy_level, block_size, N, K, components_a,
                                    M, has_weight_idx_indirect, y)) {
    return ApplyDP4AMatrixMatMulNBits(a, b, scales, zero_points, bias, batch_count, M, dispatch_M, N, K, block_size, zero_blocks_per_col, kMinMForTileOptimization, static_cast<uint32_t>(nbits), context, y, weight_index, weight_index_indirect);
  }

  // WideTileProgram
  // This program is optimized for Block32 prefill.
  const bool use_wide_tile_program = CanApplyWideTileMatMulNBits(M,
                                                                 K,
                                                                 block_size,
                                                                 nbits,
                                                                 has_weight_idx_indirect,
                                                                 components_a,
                                                                 components_b);

  const bool acc_f32 = context.EnableMatmulFp32Accumulation();

  if (use_wide_tile_program) {
    // Enforce output components to 1.
    components = 1;

    constexpr uint32_t workgroup_size = 128;
    constexpr uint32_t tile_n = workgroup_size;
    const bool is_f16 = a->DataType() == DataTypeImpl::GetType<MLFloat16>();
    const uint32_t tile_m = is_f16 ? 2 * workgroup_size / 8 : workgroup_size / 8;
    const uint32_t num_N_tile = CeilDiv(N, tile_n);
    const uint32_t num_M_tile = CeilDiv(dispatch_M, tile_m);

    bool is_nvidia = context.AdapterInfo().vendor == std::string_view{"nvidia"};

    // The template shuffles tile_m lanes in bands of subgroup_min_size when that's a
    // width it supports; otherwise (including when Subgroups isn't supported) it falls
    // back to a direct reduction.
    // NOTE: NVIDIA GPUs prefer direct reduction.
    const uint32_t subgroup_min_size = (context.HasFeature(wgpu::FeatureName::Subgroups) && !is_nvidia)
                                           ? context.AdapterInfo().subgroupMinSize
                                           : 0u;

    MatMulNBitsWideTileProgram program{has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, tile_m, tile_n, static_cast<uint32_t>(nbits), subgroup_min_size, acc_f32};
    program.SetWorkgroupSize(workgroup_size);
    program.SetDispatchGroupSize(num_N_tile, num_M_tile, batch_count);

    constexpr uint32_t kU32Components = 4;
    const uint32_t components_b_with_u32 = components_b * kU32Components;
    const uint32_t K_of_b = n_blocks_per_col * blob_size / components_b_with_u32;
    const uint32_t K_of_a = K / components_a;

    program.AddInput({a,
                      ProgramTensorMetadataDependency::TypeAndRank,
                      onnxruntime::narrow<int>(components_a)});
    program.AddInput({b,
                      ProgramTensorMetadataDependency::TypeAndRank,
                      onnxruntime::narrow<int>(components_b_with_u32)});
    program.AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_zero_points) {
      program.AddInput({zero_points,
                        ProgramTensorMetadataDependency::TypeAndRank,
                        {CeilDiv(zero_points->Shape().Size(), static_cast<int64_t>(4))},
                        4});
    }
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    if (has_weight_idx_indirect) {
      program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
    }
    program.AddOutput({y,
                       ProgramTensorMetadataDependency::TypeAndRank,
                       onnxruntime::narrow<int>(components)});
    program.AddUniformVariables({{batch_count},
                                 {dispatch_M},
                                 {N},
                                 {K_of_a},
                                 {K_of_b},
                                 {n_blocks_per_col},
                                 {zero_blocks_per_col},
                                 {num_N_tile},
                                 {num_M_tile},
                                 {weight_index}});
    program.CacheHint(nbits, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, subgroup_min_size, acc_f32);

    return context.RunProgram(program);
  }

  // Use tile_size_k_vec=32 by default for better K-dimension parallelism.
  // Intel devices use 16 as they have different subgroup/cache characteristics.
  const uint32_t tile_size_k_vec =
      (context.AdapterInfo().vendor == std::string_view{"intel"}) ? 16u : 32u;

  constexpr uint32_t workgroup_size = 128;
  constexpr uint32_t tile_size = 4;
  constexpr uint32_t kU32Components = 4;
  uint32_t components_b_with_u32 = components_b * kU32Components;
  uint32_t K_of_b = (n_blocks_per_col * blob_size) / components_b_with_u32;
  DenseBlock32Program program{tile_size, static_cast<uint32_t>(nbits), has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, single_scale_weights, tile_size_k_vec, broadcast_a, acc_f32};
  program.SetWorkgroupSize(workgroup_size);
  uint32_t num_N_tile = (N + tile_size - 1) / tile_size;
  program.SetDispatchGroupSize(num_N_tile, dispatch_M, batch_count);
  program
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                  {scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariables({{M},
                            {N},
                            {K},
                            {K / components_a},
                            {K_of_b},
                            {block_size},
                            {n_blocks_per_col},
                            {zero_blocks_per_col},
                            {num_N_tile},
                            {batch_count},
                            {weight_index},
                            {dispatch_M}})
      .CacheHint(nbits, has_zero_points, single_scale_weights, has_bias, has_weight_idx, has_weight_idx_indirect, tile_size_k_vec, broadcast_a, acc_f32);
  if (has_zero_points) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_weight_idx_indirect) {
    program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
  }
  return context.RunProgram(program);
}

}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
