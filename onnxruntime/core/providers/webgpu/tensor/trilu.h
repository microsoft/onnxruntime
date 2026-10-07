// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_TRILU_PROGRAM_CONFIG(F) F(bool, upper_)

struct TriluProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_TRILU_PROGRAM_CONFIG);
    Config(bool upper) : upper_{upper} {}
  };
  static constexpr std::string_view name = "Trilu";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"matrix_h", ProgramUniformVariableDataType::Uint32},
                                          {"matrix_w", ProgramUniformVariableDataType::Uint32},
                                          {"k", ProgramUniformVariableDataType::Int32});
};
#undef WEBGPU_TRILU_PROGRAM_CONFIG

using TriluProgram = ConfiguredProgram<TriluProgramShader>;

class Trilu final : public WebGpuKernel {
 public:
  explicit Trilu(const OpKernelInfo& info)
      : WebGpuKernel(info), upper_(info.GetAttrOrDefault<int64_t>("upper", 1) != 0) {}

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  bool upper_;
};

}  // namespace webgpu
}  // namespace onnxruntime
