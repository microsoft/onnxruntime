// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_HYPER_CONNECTION_PRE_MIX_PROGRAM_CONFIG(F) F(int, gate_layout_)

struct HyperConnectionPreMixProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_HYPER_CONNECTION_PRE_MIX_PROGRAM_CONFIG);
  static Config MakeConfig(int gate_layout) {
    Config result{};
    result.gate_layout_ = int(gate_layout);

    return result;
  }
  static constexpr std::string_view name = "HyperConnectionPreMix";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_HYPER_CONNECTION_PRE_MIX_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, count)                                                 \
  U(Uint32, branches)                                              \
  U(Uint32, hidden)                                                \
  U(Float32, reduction_scale)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_HYPER_CONNECTION_PRE_MIX_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_HYPER_CONNECTION_PRE_MIX_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_HYPER_CONNECTION_PRE_MIX_PROGRAM_CONFIG

using HyperConnectionPreMixProgram = ConfiguredProgram<HyperConnectionPreMixProgramShader>;

class HyperConnectionPreMix final : public WebGpuKernel {
 public:
  explicit HyperConnectionPreMix(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t num_branches_;
  float reduction_scale_;
};

}  // namespace onnxruntime::contrib::webgpu
