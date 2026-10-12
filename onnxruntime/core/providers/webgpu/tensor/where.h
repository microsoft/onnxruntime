// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_WHERE_PROGRAM_CONFIG(F) \
  F(bool, is_broadcast_)               \
  F(bool, is_int64_)

struct WhereProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_WHERE_PROGRAM_CONFIG);
  static Config MakeConfig(bool is_broadcast, bool is_int64 = false) {
    Config result{};
    result.is_broadcast_ = bool{is_broadcast};
    result.is_int64_ = bool{is_int64};

    return result;
  }
  static constexpr std::string_view name = "Where";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_WHERE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, vec_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_WHERE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_WHERE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_WHERE_PROGRAM_CONFIG

using WhereProgram = ConfiguredProgram<WhereProgramShader>;

class Where final : public WebGpuKernel {
 public:
  Where(const OpKernelInfo& info) : WebGpuKernel{info} {
  }

  Status ComputeInternal(ComputeContext& context) const override;
};

// Factory functions for conditional int64 support (registered via RegisterKernels).
KernelCreateInfo CreateWhereVersionedKernelInfo(int start_version, int end_version, bool enable_int64);
KernelCreateInfo CreateWhereKernelInfo(int since_version, bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
