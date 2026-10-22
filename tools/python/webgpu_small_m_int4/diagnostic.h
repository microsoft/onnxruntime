// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdlib>
#include <initializer_list>
#include <string>
#include <string_view>

#include "core/common/safeint.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime::contrib::webgpu {
namespace {

uint32_t DiagnosticTileSize(const char* name, uint32_t fallback, std::initializer_list<uint32_t> supported) {
  const char* text = std::getenv(name);
  if (text == nullptr) {
    return fallback;
  }
  for (uint32_t value : supported) {
    if (std::string_view{text} == std::to_string(value)) {
      return value;
    }
  }
  ORT_THROW("Unsupported diagnostic tile setting: ", name, "=", text);
}

class DiagnosticSmallMSimtProgram final : public Program<DiagnosticSmallMSimtProgram> {
 public:
  DiagnosticSmallMSimtProgram(uint32_t bm, uint32_t bn, uint32_t bk, uint32_t a_components,
                              uint32_t a_block_padding, bool has_zero_points, bool has_bias, bool single_scale)
      : Program{"DiagnosticSmallMSimt"}, bm_{bm}, bn_{bn}, bk_{bk}, a_components_{a_components}, a_block_padding_{a_block_padding}, has_zero_points_{has_zero_points}, has_bias_{has_bias}, single_scale_{single_scale} {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DiagnosticSmallMSimtProgram);

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias);
    const ShaderVariableHelper* zero_points = nullptr;
    const ShaderVariableHelper* bias = nullptr;
    if (has_zero_points_) {
      zero_points = &shader.AddInput("zero_points", ShaderUsage::UseValueTypeAlias);
    }
    if (has_bias_) {
      bias = &shader.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    const uint32_t k_lanes = 128 / bn_;
    const uint32_t a_block_stride = 8 + a_block_padding_;
    const uint32_t a_row_stride = bk_ / 32 * a_block_stride;
    auto& impl = shader.AdditionalImplementation();
    impl << "const BM = " << bm_ << "u;\nconst BN = " << bn_ << "u;\nconst BK = " << bk_
         << "u;\nconst K_LANES = " << k_lanes << "u;\n"
         << "var<workgroup> a_stage: array<vec4<f16>, " << bm_ * a_row_stride << ">;\n"
         << "var<workgroup> partial: array<f32, " << bm_ * 128 << ">;\n";

    auto& body = shader.MainFunctionBody();
    body << R"(
  if (workgroup_idx >= uniforms.num_n_tile * uniforms.batch_count) { return; }
  let batch = workgroup_idx / uniforms.num_n_tile;
  let n = (workgroup_idx % uniforms.num_n_tile) * BN + local_idx / K_LANES;
  let k_lane = local_idx % K_LANES;
  let b_base = uniforms.weight_index * uniforms.N * uniforms.b_chunks;
)";
    if (single_scale_) {
      body << "  let scale_base = uniforms.weight_index;\n";
    } else {
      body << "  let scale_base = uniforms.weight_index * uniforms.N * uniforms.blocks;\n";
    }
    // Emit distinct scalars so array indexing cannot put the live row accumulators in private memory.
    for (uint32_t row = 0; row < bm_; ++row) {
      body << "  var acc" << row << " = 0.0f;\n";
    }
    body << R"(
  for (var kb = 0u; kb < uniforms.K; kb += BK) {
    for (var i = local_idx; i < BM * (BK / 4u); i += 128u) {
      let row = i / (BK / 4u);
      let k = kb + (i % (BK / 4u)) * 4u;
      var value = vec4<f16>(0.0h);
)";
    if (a_components_ == 4) {
      body << "      if (row < uniforms.M && k < uniforms.K) {\n"
           << "        value = " << a.GetByOffset("(batch * uniforms.M + row) * (uniforms.K / 4u) + k / 4u")
           << ";\n      }\n";
    } else {
      body << "      if (row < uniforms.M) {\n";
      for (uint32_t component = 0; component < 4; ++component) {
        const std::string k = "k + " + std::to_string(component) + "u";
        body << "        if (" << k << " < uniforms.K) { value[" << component << "] = "
             << a.GetByOffset("(batch * uniforms.M + row) * uniforms.K + " + k) << "; }\n";
      }
      body << "      }\n";
    }
    if (a_block_padding_ == 0) {
      body << "      a_stage[i] = value;\n";
    } else {
      // An extra vec4 per block32 breaks the 64-byte stride between logical K lanes.
      body << "      a_stage[row * " << a_row_stride << "u + (i % (BK / 4u)) / 8u * "
           << a_block_stride << "u + i % 8u] = value;\n";
    }
    body << R"(
    }
    workgroupBarrier();
    for (var block = k_lane; block < BK / 32u; block += K_LANES) {
      let k = kb + block * 32u;
      if (n < uniforms.N && k < uniforms.K) {
        let scale_block = k / uniforms.block_size;
)";
    body << "        let scale = "
         << scales.GetByOffset(single_scale_ ? "scale_base" : "scale_base + n * uniforms.blocks + scale_block")
         << ";\n";
    if (has_zero_points_) {
      body << "        let zero_index = uniforms.weight_index * uniforms.N * uniforms.zero_stride";
      if (!single_scale_) {
        body << " + n * uniforms.zero_stride + scale_block";
      }
      body << ";\n        let zero_word = " << zero_points->GetByOffset("zero_index / 8u") << ";\n"
           << "        let zero = f16((zero_word >> ((zero_index % 8u) * 4u)) & 15u);\n";
    } else {
      body << "        let zero = 8.0h;\n";
    }
    body << "        let packed = " << b.GetByOffset("b_base + n * uniforms.b_chunks + k / 32u") << ";\n";
    for (uint32_t word = 0; word < 4; ++word) {
      body << "        {\n          let bits = packed[" << word << "];\n"
           << "          let lo = vec4<f16>(unpack4xU8(bits & 0x0F0F0F0Fu)) - vec4<f16>(zero);\n"
           << "          let hi = vec4<f16>(unpack4xU8((bits >> 4u) & 0x0F0F0F0Fu)) - vec4<f16>(zero);\n"
           << "          let w0_half = vec4<f16>(lo.x, hi.x, lo.y, hi.y) * scale;\n"
           << "          let w1_half = vec4<f16>(lo.z, hi.z, lo.w, hi.w) * scale;\n"
           << "          let w0 = vec4<f32>(w0_half);\n          let w1 = vec4<f32>(w1_half);\n";
      for (uint32_t row = 0; row < bm_; ++row) {
        const uint32_t offset = row * a_row_stride + word * 2;
        body << "          acc" << row << " += dot(vec4<f32>(a_stage[block * " << a_block_stride << "u + " << offset
             << "u]), w0) + dot(vec4<f32>(a_stage[block * " << a_block_stride << "u + " << offset + 1 << "u]), w1);\n";
      }
      body << "        }\n";
    }
    body << R"(
      }
    }
    workgroupBarrier();
  }
)";
    for (uint32_t row = 0; row < bm_; ++row) {
      body << "  partial[" << row * 128 << "u + local_idx] = acc" << row << ";\n";
    }
    body << R"(
  workgroupBarrier();
  for (var i = local_idx; i < uniforms.M * BN; i += 128u) {
    let row = i / BN;
    let col = (workgroup_idx % uniforms.num_n_tile) * BN + i % BN;
    if (col < uniforms.N) {
      var sum = 0.0f;
      for (var lane = 0u; lane < K_LANES; lane++) {
        sum += partial[row * 128u + (i % BN) * K_LANES + lane];
      }
)";
    if (has_bias_) {
      body << "      sum += f32(" << bias->GetByOffset("uniforms.weight_index * uniforms.N + col") << ");\n";
    }
    body << "      "
         << output.SetByOffset("(batch * uniforms.M + row) * uniforms.N + col", "f16(sum)")
         << "\n    }\n  }\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"blocks", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"b_chunks", ProgramUniformVariableDataType::Uint32},
      {"zero_stride", ProgramUniformVariableDataType::Uint32},
      {"num_n_tile", ProgramUniformVariableDataType::Uint32},
      {"batch_count", ProgramUniformVariableDataType::Uint32},
      {"weight_index", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t bm_;
  uint32_t bn_;
  uint32_t bk_;
  uint32_t a_components_;
  uint32_t a_block_padding_;
  bool has_zero_points_;
  bool has_bias_;
  bool single_scale_;
};

Status RunDiagnosticSmallMSimt(const Tensor* a, const Tensor* b, const Tensor* scales,
                               const Tensor* zero_points, const Tensor* bias, uint32_t M, uint32_t N,
                               uint32_t K, uint32_t block_size, uint32_t batch_count, uint32_t weight_index,
                               const Tensor* weight_index_indirect, uint32_t dispatch_M, uint32_t nbits,
                               onnxruntime::webgpu::ComputeContext& context, Tensor* y) {
  ORT_RETURN_IF_NOT(nbits == 4 && M >= 1 && M <= 8 && N > 0 && K > 0,
                    "SIMT diagnostic requires INT4, 1 <= effective M <= 8, and positive N/K.");
  ORT_RETURN_IF_NOT(context.HasFeature(wgpu::FeatureName::ShaderF16) &&
                        a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
                        scales->DataType() == a->DataType() && y->DataType() == a->DataType(),
                    "SIMT diagnostic requires shader-f16 and FP16 activations/scales/output.");
  ORT_RETURN_IF_NOT(b->DataType() == DataTypeImpl::GetType<uint8_t>() &&
                        (zero_points == nullptr || zero_points->DataType() == DataTypeImpl::GetType<uint8_t>()) &&
                        (bias == nullptr || bias->DataType() == a->DataType()),
                    "SIMT diagnostic requires uint8 packed B/zero points and FP16 bias.");
  ORT_RETURN_IF_NOT(weight_index_indirect == nullptr && dispatch_M == M,
                    "SIMT diagnostic does not support per-row weight selection or activation-row replay.");

  const bool single_scale = static_cast<uint64_t>(block_size) == static_cast<uint64_t>(K) * N;
  const uint32_t block_size_per_col = single_scale ? K : block_size;
  ORT_RETURN_IF_NOT(block_size_per_col >= 32 && block_size_per_col % 32 == 0,
                    "SIMT diagnostic requires block sizes divisible by 32.");
  const uint32_t blocks = CeilDiv(K, block_size_per_col);
  const uint32_t b_chunks = onnxruntime::narrow<uint32_t>(SafeInt<uint64_t>(blocks) * block_size_per_col / 32);
  const uint32_t zero_stride = CeilDiv(blocks, 2u) * 2u;
  const uint64_t matrix_count = static_cast<uint64_t>(weight_index) + 1;
  const uint64_t scale_count =
      single_scale ? matrix_count : static_cast<uint64_t>(SafeInt<uint64_t>(matrix_count) * N * blocks);
  ORT_RETURN_IF_NOT(static_cast<uint64_t>(b->Shape().Size()) >= SafeInt<uint64_t>(matrix_count) * N * b_chunks * 16 &&
                        static_cast<uint64_t>(scales->Shape().Size()) >= scale_count,
                    "SIMT diagnostic packed B/scales do not match the declared quantization layout.");
  ORT_RETURN_IF_NOT(zero_points == nullptr ||
                        static_cast<uint64_t>(zero_points->Shape().Size()) >=
                            SafeInt<uint64_t>(matrix_count) * N * zero_stride / 2,
                    "SIMT diagnostic zero points do not include per-column byte padding.");
  ORT_RETURN_IF_NOT(bias == nullptr ||
                        static_cast<uint64_t>(bias->Shape().Size()) >= SafeInt<uint64_t>(matrix_count) * N,
                    "SIMT diagnostic bias is too short.");

  const uint32_t bm = M <= 2 ? M : M <= 4 ? 4
                                          : 8;
  const uint32_t bn = DiagnosticTileSize("ORT_WEBGPU_SMALL_M_BN", 16, {8, 16, 32});
  const uint32_t bk = DiagnosticTileSize("ORT_WEBGPU_SMALL_M_BK", 512, {256, 512, 1024});
  const uint32_t a_block_padding = DiagnosticTileSize("ORT_WEBGPU_SMALL_M_A_BLOCK_PADDING", 0, {0, 1});
  const uint32_t shared_bytes = bm * (bk / 32) * (8 + a_block_padding) * 8 + bm * 128 * 4;
  ORT_RETURN_IF_NOT(shared_bytes <= context.DeviceLimits().maxComputeWorkgroupStorageSize,
                    "SIMT diagnostic workgroup storage exceeds the device limit: ", shared_bytes);
  const uint32_t a_components = K % 4 == 0 ? 4 : 1;
  const uint32_t n_tiles = CeilDiv(N, bn);
  DiagnosticSmallMSimtProgram program{bm, bn, bk, a_components, a_block_padding,
                                      zero_points != nullptr, bias != nullptr, single_scale};
  program.SetWorkgroupSize(128);
  program.SetDispatchGroupSize(n_tiles, batch_count);
  program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(a_components)});
  program.AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16});
  program.AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
  if (zero_points != nullptr) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {CeilDiv(zero_points->Shape().Size(), int64_t{4})}, 4});
  }
  if (bias != nullptr) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  program.AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
  program.AddUniformVariables({M, N, K, blocks, block_size_per_col, b_chunks, zero_stride,
                               n_tiles, batch_count, weight_index});
  program.CacheHint(bm, bn, bk, a_components, a_block_padding, zero_points != nullptr, bias != nullptr, single_scale);
  return context.RunProgram(program);
}

}  // namespace
}  // namespace onnxruntime::contrib::webgpu
