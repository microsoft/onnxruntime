// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class ComputeChannelStatsProgram final : public Program<ComputeChannelStatsProgram> {
 public:
  explicit ComputeChannelStatsProgram(int components)
      : Program{"ComputeChannelStats"}, components_(components) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  int components_;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"epsilon", ProgramUniformVariableDataType::Float32});
};

class InstanceNormProgram final : public Program<InstanceNormProgram> {
 public:
  InstanceNormProgram() : Program{"InstanceNorm"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});
};

class InstanceNormProgramNHWC final : public Program<InstanceNormProgramNHWC> {
 public:
  InstanceNormProgramNHWC(int components) : Program{"InstanceNormNHWC"}, components_(components) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32}, {"components", ProgramUniformVariableDataType::Uint32}, {"C", ProgramUniformVariableDataType::Uint32}, {"H", ProgramUniformVariableDataType::Uint32});

 private:
  int components_;
};

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
Status ComputeChannelStats(ComputeContext& context, const Tensor* input, float epsilon, Tensor* output);

}  // namespace webgpu
}  // namespace onnxruntime
