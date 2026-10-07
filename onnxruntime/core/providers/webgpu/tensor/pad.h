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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PAD_PROGRAM_CONFIG);
    Config(const Mode mode, bool dim_value_zero, bool is_float16)
        : mode_{mode}, dim_value_zero_{dim_value_zero}, is_float16_{is_float16} {}
  };
  static constexpr std::string_view name = "Pad";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"lower_pads", ProgramUniformVariableDataType::Int32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"constant_value", ProgramUniformVariableDataType::Uint32});
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
