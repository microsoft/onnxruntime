// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <optional>

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

// TODO: Move this common subgroup implementation out of vendor/intel. Keep only
// Intel-specific automatic-selection thresholds and tuning under the vendor directory.
#define WEBGPU_MAT_MUL_SUBGROUP_PROGRAM_CONFIG(F) \
  F(ShaderActivation, activation_)                \
  F(bool, has_bias_)                              \
  F(bool, is_vec4_)                               \
  F(bool, a_vec4_)                                \
  F(bool, b_is_fp16_)                             \
  F(bool, use_f32_accumulation_)                  \
  F(bool, is_channels_last_)                      \
  F(InlinedVector<int64_t>, elements_per_thread_)

struct MatMulSubgroupProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_MAT_MUL_SUBGROUP_PROGRAM_CONFIG);
  static Config MakeConfig(const Activation& activation,
                           bool bias,
                           bool is_vec4,
                           bool a_vec4,
                           bool b_is_fp16,
                           bool use_f32_accumulation,
                           bool is_channels_last,
                           const gsl::span<int64_t>& elements_per_thread) {
    Config result{};
    result.activation_ = MakeShaderActivation(activation);
    result.has_bias_ = bool{bias};
    result.is_vec4_ = bool{is_vec4};
    result.a_vec4_ = bool{a_vec4};
    result.b_is_fp16_ = bool{b_is_fp16};
    result.use_f32_accumulation_ = bool{use_f32_accumulation};
    result.is_channels_last_ = bool{is_channels_last};
    result.elements_per_thread_ = InlinedVector<int64_t>(elements_per_thread.begin(), elements_per_thread.end());

    return result;
  }
  static constexpr std::string_view name = "MatMulSubgroup";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_MAT_MUL_SUBGROUP_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, dim_a_outer)                                   \
  U(Uint32, dim_b_outer)                                   \
  U(Uint32, dim_inner)                                     \
  U(Uint32, logical_dispatch_x)                            \
  U(Uint32, logical_dispatch_y)                            \
  U(Uint32, logical_dispatch_z)                            \
  U(Float32, activation_param_0)                           \
  U(Float32, activation_param_1)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_MAT_MUL_SUBGROUP_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_MAT_MUL_SUBGROUP_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_MAT_MUL_SUBGROUP_PROGRAM_CONFIG

using MatMulSubgroupProgram = ConfiguredProgram<MatMulSubgroupProgramShader>;

std::optional<uint32_t> SelectMatMulSubgroupSize(const ComputeContext& context);

Status ApplyMatMulSubgroup(ComputeContext& context,
                           const Activation& activation,
                           const std::vector<const Tensor*>& inputs,
                           Tensor* output,
                           bool is_channels_last,
                           uint32_t subgroup_size);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
