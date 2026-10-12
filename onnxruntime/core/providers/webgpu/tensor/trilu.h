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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TRILU_PROGRAM_CONFIG);
  static Config MakeConfig(bool upper) {
    Config result{};
    result.upper_ = bool{upper};

    return result;
  }
  static constexpr std::string_view name = "Trilu";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_TRILU_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                        \
  U(Uint32, matrix_h)                           \
  U(Uint32, matrix_w)                           \
  U(Int32, k)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_TRILU_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_TRILU_PROGRAM_SHADER_UNIFORMS
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
