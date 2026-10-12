// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_HYPER_CONNECTION_POST_MIX_PROGRAM_CONFIG(F) \
  F(int, gate_layout_)                                     \
  F(bool, has_stream_mix_)

struct HyperConnectionPostMixProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_HYPER_CONNECTION_POST_MIX_PROGRAM_CONFIG);
  static Config MakeConfig(int gate_layout, bool has_stream_mix) {
    Config result{};
    result.gate_layout_ = int(gate_layout);
    result.has_stream_mix_ = bool(has_stream_mix);

    return result;
  }
  static constexpr std::string_view name = "HyperConnectionPostMix";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_HYPER_CONNECTION_POST_MIX_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, count)                                                  \
  U(Uint32, branches)                                               \
  U(Uint32, hidden)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_HYPER_CONNECTION_POST_MIX_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_HYPER_CONNECTION_POST_MIX_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_HYPER_CONNECTION_POST_MIX_PROGRAM_CONFIG

using HyperConnectionPostMixProgram = ConfiguredProgram<HyperConnectionPostMixProgramShader>;

class HyperConnectionPostMix final : public WebGpuKernel {
 public:
  explicit HyperConnectionPostMix(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t num_branches_;
};

}  // namespace onnxruntime::contrib::webgpu
