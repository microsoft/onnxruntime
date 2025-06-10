// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/tensor/padbase.h"

namespace onnxruntime {
namespace webgpu {

class PadProgram final : public Program<PadProgram> {
 public:
  PadProgram(const Mode mode, bool dim_value_zero, bool is_float16) : Program<PadProgram>{"Pad"},
                                                                      mode_{mode},
                                                                      dim_value_zero_{dim_value_zero},
                                                                      is_float16_{is_float16} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"lower_pads", ProgramUniformVariableDataType::Int32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"constant_value", ProgramUniformVariableDataType::Uint32});

 private:
  Mode mode_;
  bool dim_value_zero_;
  bool is_float16_;
};

class Pad final : public PadBase, public WebGpuKernel {
 public:
  Pad(const OpKernelInfo& info) : PadBase(info), WebGpuKernel(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
