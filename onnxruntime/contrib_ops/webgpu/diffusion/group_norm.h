// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using onnxruntime::webgpu::ComputeContext;
using onnxruntime::webgpu::ConfiguredProgram;
using onnxruntime::webgpu::ConfiguredShaderHelper;
using onnxruntime::webgpu::ProgramUniformVariableDataType;
using onnxruntime::webgpu::WebGpuKernel;

// Computes per-(batch, group) mean and inverse standard deviation of x (+ skip + bias).
// One workgroup per (batch, group); output is [N * G] with 2 components (mean, inv_std) in f32.
#define WEBGPU_GROUP_NORM_STATS_PROGRAM_CONFIG(F) \
  F(int, components_)                             \
  F(uint32_t, workgroup_size_)                    \
  F(bool, has_skip_)                              \
  F(bool, skip_broadcast_)                        \
  F(bool, has_bias_)

struct GroupNormStatsProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GROUP_NORM_STATS_PROGRAM_CONFIG);
  static Config MakeConfig(int components, uint32_t workgroup_size, bool has_skip, bool skip_broadcast, bool has_bias) {
    Config result{};
    result.components_ = int{components};
    result.workgroup_size_ = uint32_t{workgroup_size};
    result.has_skip_ = bool{has_skip};
    result.skip_broadcast_ = bool{skip_broadcast};
    result.has_bias_ = bool{has_bias};

    return result;
  }
  static constexpr std::string_view name = "GroupNormStats";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GROUP_NORM_STATS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, hw)                                            \
  U(Uint32, c_comp)                                        \
  U(Uint32, cg_comp)                                       \
  U(Uint32, groups)                                        \
  U(Float32, epsilon)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GROUP_NORM_STATS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GROUP_NORM_STATS_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GROUP_NORM_STATS_PROGRAM_CONFIG

using GroupNormStatsProgram = ConfiguredProgram<GroupNormStatsProgramShader>;

// Applies normalization with per-channel affine (gamma, beta) and optional SiLU activation.
// With skip: normalizes (x + skip + bias) and optionally writes the sum to the S output.
#define WEBGPU_GROUP_NORM_APPLY_PROGRAM_CONFIG(F) \
  F(int, components_)                             \
  F(bool, use_silu_)                              \
  F(bool, has_skip_)                              \
  F(bool, skip_broadcast_)                        \
  F(bool, has_bias_)                              \
  F(bool, has_sum_output_)

struct GroupNormApplyProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GROUP_NORM_APPLY_PROGRAM_CONFIG);
  static Config MakeConfig(int components, bool use_silu, bool has_skip, bool skip_broadcast, bool has_bias, bool has_sum_output) {
    Config result{};
    result.components_ = int{components};
    result.use_silu_ = bool{use_silu};
    result.has_skip_ = bool{has_skip};
    result.skip_broadcast_ = bool{skip_broadcast};
    result.has_bias_ = bool{has_bias};
    result.has_sum_output_ = bool{has_sum_output};

    return result;
  }
  static constexpr std::string_view name = "GroupNormApply";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GROUP_NORM_APPLY_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                   \
  U(Uint32, hw)                                            \
  U(Uint32, c_comp)                                        \
  U(Uint32, cg_comp)                                       \
  U(Uint32, groups)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GROUP_NORM_APPLY_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GROUP_NORM_APPLY_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GROUP_NORM_APPLY_PROGRAM_CONFIG

using GroupNormApplyProgram = ConfiguredProgram<GroupNormApplyProgramShader>;

// Handles both com.microsoft.GroupNorm and com.microsoft.SkipGroupNorm (channels_last only).
class GroupNorm final : public WebGpuKernel {
 public:
  GroupNorm(const OpKernelInfo& info) : WebGpuKernel(info) {
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);
    ORT_ENFORCE(epsilon_ >= 0.0f, "epsilon must be non-negative, got ", epsilon_);

    ORT_ENFORCE(info.GetAttr("groups", &groups_).IsOK(), "groups attribute is required");
    ORT_ENFORCE(groups_ > 0, "groups must be positive, got ", groups_);

    ORT_ENFORCE(info.GetAttr("activation", &activation_).IsOK(), "activation attribute is required");
    ORT_ENFORCE(activation_ == 0 || activation_ == 1, "activation must be 0 (None) or 1 (SiLU), got ", activation_);

    channels_last_ = info.GetAttrOrDefault<int64_t>("channels_last", 1);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float epsilon_;
  int64_t groups_;
  int64_t activation_;
  int64_t channels_last_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
