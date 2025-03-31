// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

template <typename T>
class Range : public WebGpuKernel {
 public:
  explicit Range(const OpKernelInfo& info) : WebGpuKernel(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

class RangeProgram : public Program<RangeProgram> {
 public:
  RangeProgram() : Program{"Range"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"start", ProgramUniformVariableDataType::Uint32},
                                          {"delta", ProgramUniformVariableDataType::Uint32});
};

}  // namespace webgpu
}  // namespace onnxruntime
