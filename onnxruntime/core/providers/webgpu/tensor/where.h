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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_WHERE_PROGRAM_CONFIG);
    Config(bool is_broadcast, bool is_int64 = false) : is_broadcast_{is_broadcast}, is_int64_{is_int64} {}
  };
  static constexpr std::string_view name = "Where";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32});
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
