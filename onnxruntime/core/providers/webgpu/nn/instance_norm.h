// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_CHANNEL_SCALE_CONFIG(F) \
  F(float, epsilon)

WEBGPU_DECLARE_SPECIALIZATION(ChannelScaleConfig, WEBGPU_CHANNEL_SCALE_CONFIG);
#undef WEBGPU_CHANNEL_SCALE_CONFIG

struct ComputeChannelScaleShiftShader {
  using Config = ChannelScaleConfig;
  static constexpr std::string_view name = "ComputeChannelScaleShift";
  static Status GenerateShaderCode(const Config& config, ConfiguredShaderHelper& sh);
};

using ComputeChannelScaleShiftProgram = ConfiguredProgram<ComputeChannelScaleShiftShader>;

#define WEBGPU_INSTANCE_NORM_PROGRAM_CONFIG(F)

struct InstanceNormProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_INSTANCE_NORM_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "InstanceNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_INSTANCE_NORM_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_INSTANCE_NORM_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_INSTANCE_NORM_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_INSTANCE_NORM_PROGRAM_CONFIG

using InstanceNormProgram = ConfiguredProgram<InstanceNormProgramShader>;

#define WEBGPU_INSTANCE_NORM_PROGRAM_N_H_W_C_CONFIG(F) F(int, components_)

struct InstanceNormProgramNHWCShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_INSTANCE_NORM_PROGRAM_N_H_W_C_CONFIG);
  static Config MakeConfig(int components) {
    Config result{};
    result.components_ = int(components);

    return result;
  }
  static constexpr std::string_view name = "InstanceNormNHWC";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_INSTANCE_NORM_PROGRAM_N_H_W_C_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                        \
  U(Uint32, components)                                         \
  U(Uint32, C)                                                  \
  U(Uint32, H)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_INSTANCE_NORM_PROGRAM_N_H_W_C_SHADER_UNIFORMS);
#undef WEBGPU_INSTANCE_NORM_PROGRAM_N_H_W_C_SHADER_UNIFORMS
};
#undef WEBGPU_INSTANCE_NORM_PROGRAM_N_H_W_C_CONFIG

using InstanceNormProgramNHWC = ConfiguredProgram<InstanceNormProgramNHWCShader>;

template <bool is_nhwc>
class InstanceNorm final : public WebGpuKernel {
 public:
  InstanceNorm(const OpKernelInfo& info) : WebGpuKernel(info) {
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);
  }
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float epsilon_;
};
Status ComputeChannelScaleAndShift(ComputeContext& context, const Tensor* input, const Tensor* scale, const Tensor* bias, float epsilon, Tensor* output);

}  // namespace webgpu
}  // namespace onnxruntime
