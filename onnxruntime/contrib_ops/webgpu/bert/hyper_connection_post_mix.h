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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_HYPER_CONNECTION_POST_MIX_PROGRAM_CONFIG);
    Config(int gate_layout, bool has_stream_mix) : gate_layout_(gate_layout), has_stream_mix_(has_stream_mix) {}
  };
  static constexpr std::string_view name = "HyperConnectionPostMix";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"hidden", ProgramUniformVariableDataType::Uint32});
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
