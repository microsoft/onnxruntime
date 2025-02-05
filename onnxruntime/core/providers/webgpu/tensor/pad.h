// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/tensor/padbase.h"

namespace onnxruntime {
namespace webgpu {

template <typename T>
class PadProgram final : public Program<PadProgram<T> > {
 public:
  PadProgram(const Mode mode, bool dim_value_zero) : Program<PadProgram<T> >{"Pad"}, mode_{mode}, dim_value_zero_{dim_value_zero} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"lower_pads", ProgramUniformVariableDataType::Int32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"constant_value",
                                           std::is_same_v<T, float> ? ProgramUniformVariableDataType::Float32 : (std::is_same_v<T, int32_t> ? ProgramUniformVariableDataType::Int32 : (std::is_same_v<T, uint32_t> ? ProgramUniformVariableDataType::Uint32 : ProgramUniformVariableDataType::Float16))});

 private:
  Mode mode_;
  bool dim_value_zero_;
};

template <typename T>
class Pad final : public PadBase, public WebGpuKernel {
 public:
  Pad(const OpKernelInfo& info) : PadBase(info), WebGpuKernel(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
