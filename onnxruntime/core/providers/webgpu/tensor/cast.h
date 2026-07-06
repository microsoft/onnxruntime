// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class CastProgram final : public Program<CastProgram> {
 public:
  CastProgram(int32_t to, bool is_from_int64, bool is_from_float, bool is_from_unsigned)
      : Program{"Cast"}, to_{to}, is_from_int64_{is_from_int64}, is_from_float_{is_from_float}, is_from_unsigned_{is_from_unsigned} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  int32_t to_;
  bool is_from_int64_;
  bool is_from_float_;
  bool is_from_unsigned_;
};

class Cast final : public WebGpuKernel {
 public:
  Cast(const OpKernelInfo& info) : WebGpuKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = onnxruntime::narrow<int32_t>(to);

    // ignore attribute 'saturate' as float8 is not supported in WebGPU
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int32_t to_;
};

// Create Cast kernel info with appropriate type constraints based on int64 support
template <int StartVersion, int EndVersion = StartVersion>
KernelCreateInfo CreateCastKernelInfo(bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
