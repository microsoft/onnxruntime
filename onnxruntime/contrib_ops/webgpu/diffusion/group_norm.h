// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using onnxruntime::webgpu::ComputeContext;
using onnxruntime::webgpu::Program;
using onnxruntime::webgpu::ProgramUniformVariableDataType;
using onnxruntime::webgpu::ShaderHelper;
using onnxruntime::webgpu::WebGpuKernel;

// Computes per-(batch, group) mean and inverse standard deviation of x (+ skip + bias).
// One workgroup per (batch, group); output is [N * G] with 2 components (mean, inv_std) in f32.
class GroupNormStatsProgram final : public Program<GroupNormStatsProgram> {
 public:
  GroupNormStatsProgram(int components, uint32_t workgroup_size, bool has_skip, bool skip_broadcast, bool has_bias)
      : Program{"GroupNormStats"},
        components_{components},
        workgroup_size_{workgroup_size},
        has_skip_{has_skip},
        skip_broadcast_{skip_broadcast},
        has_bias_{has_bias} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hw", ProgramUniformVariableDataType::Uint32},
      {"c_comp", ProgramUniformVariableDataType::Uint32},
      {"cg_comp", ProgramUniformVariableDataType::Uint32},
      {"groups", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  int components_;
  uint32_t workgroup_size_;
  bool has_skip_;
  bool skip_broadcast_;
  bool has_bias_;
};

// Applies normalization with per-channel affine (gamma, beta) and optional SiLU activation.
// With skip: normalizes (x + skip + bias) and optionally writes the sum to the S output.
class GroupNormApplyProgram final : public Program<GroupNormApplyProgram> {
 public:
  GroupNormApplyProgram(int components, bool use_silu, bool has_skip, bool skip_broadcast, bool has_bias, bool has_sum_output)
      : Program{"GroupNormApply"},
        components_{components},
        use_silu_{use_silu},
        has_skip_{has_skip},
        skip_broadcast_{skip_broadcast},
        has_bias_{has_bias},
        has_sum_output_{has_sum_output} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"hw", ProgramUniformVariableDataType::Uint32},
      {"c_comp", ProgramUniformVariableDataType::Uint32},
      {"cg_comp", ProgramUniformVariableDataType::Uint32},
      {"groups", ProgramUniformVariableDataType::Uint32});

 private:
  int components_;
  bool use_silu_;
  bool has_skip_;
  bool skip_broadcast_;
  bool has_bias_;
  bool has_sum_output_;
};

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
