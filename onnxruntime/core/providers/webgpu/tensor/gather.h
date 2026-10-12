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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATHER_PROGRAM_CONFIG);
  static Config MakeConfig(const uint32_t axis, bool is_int64) {
    Config result{};
    result.axis_ = uint32_t{axis};
    result.is_int64_ = bool{is_int64};

    return result;
  }
  static constexpr std::string_view name = "Gather";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_GATHER_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, data_size)                           \
  U(Uint32, output_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATHER_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATHER_PROGRAM_SHADER_UNIFORMS
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
