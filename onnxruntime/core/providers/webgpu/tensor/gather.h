// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/tensor/gatherbase.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_GATHER_PROGRAM_CONFIG(F) \
  F(uint32_t, axis_)                    \
  F(bool, is_int64_)

struct GatherProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATHER_PROGRAM_CONFIG);
    Config(const uint32_t axis, bool is_int64) : axis_{axis}, is_int64_{is_int64} {}
  };
  static constexpr std::string_view name = "Gather";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"data_size", ProgramUniformVariableDataType::Uint32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_GATHER_PROGRAM_CONFIG

using GatherProgram = ConfiguredProgram<GatherProgramShader>;

class Gather final : public WebGpuKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : WebGpuKernel(info), GatherBase(info) {}

 protected:
  Status ComputeInternal(ComputeContext& context) const override;
};

// Create Gather kernel info with appropriate type constraints based on int64 support
KernelCreateInfo CreateGatherVersionedKernelInfo(int start_version, int end_version, bool enable_int64);
KernelCreateInfo CreateGatherKernelInfo(int since_version, bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
