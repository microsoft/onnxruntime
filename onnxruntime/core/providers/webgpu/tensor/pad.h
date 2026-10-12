// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/tensor/padbase.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_PAD_PROGRAM_CONFIG(F) \
  F(Mode, mode_)                     \
  F(bool, dim_value_zero_)           \
  F(bool, is_float16_)

struct PadProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_PAD_PROGRAM_CONFIG);
  static Config MakeConfig(const Mode mode, bool dim_value_zero, bool is_float16) {
    Config result{};
    result.mode_ = Mode{mode};
    result.dim_value_zero_ = bool{dim_value_zero};
    result.is_float16_ = bool{is_float16};

    return result;
  }
  static constexpr std::string_view name = "Pad";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_PAD_PROGRAM_SHADER_UNIFORMS(U) \
  U(Int32, lower_pads)                        \
  U(Uint32, output_size)                      \
  U(Uint32, constant_value)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_PAD_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_PAD_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_PAD_PROGRAM_CONFIG

using PadProgram = ConfiguredProgram<PadProgramShader>;

class Pad final : public PadBase, public WebGpuKernel {
 public:
  Pad(const OpKernelInfo& info) : PadBase(info), WebGpuKernel(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
