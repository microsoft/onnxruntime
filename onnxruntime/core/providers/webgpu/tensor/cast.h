// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/configured_program.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_CAST_PROGRAM_CONFIG(F) \
  F(int32_t, to_)                     \
  F(bool, is_from_int64_)             \
  F(bool, is_from_float_)             \
  F(bool, is_from_unsigned_)          \
  F(bool, is_from_uint8_)

struct CastProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_CAST_PROGRAM_CONFIG);
    Config(int32_t to, bool is_from_int64, bool is_from_float, bool is_from_unsigned, bool is_from_uint8)
        : to_{to},
          is_from_int64_{is_from_int64},
          is_from_float_{is_from_float},
          is_from_unsigned_{is_from_unsigned},
          is_from_uint8_{is_from_uint8} {}
  };
  static constexpr std::string_view name = "Cast";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_CAST_PROGRAM_CONFIG

using CastProgram = ConfiguredProgram<CastProgramShader>;

class Cast final : public WebGpuKernel {
 public:
  Cast(const OpKernelInfo& info) : WebGpuKernel(info) {
    int64_t to = 0;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = onnxruntime::narrow<int32_t>(to);

    // ignore attribute 'saturate' as float8 is not supported in WebGPU
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int32_t to_;
};

// Create Cast kernel info with appropriate type constraints based on int64 support.
KernelCreateInfo CreateCastVersionedKernelInfo(int start_version, int end_version, bool enable_int64);
KernelCreateInfo CreateCastKernelInfo(int since_version, bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
