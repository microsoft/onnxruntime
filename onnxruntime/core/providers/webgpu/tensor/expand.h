// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class ExpandProgram final : public Program<ExpandProgram> {
 public:
  ExpandProgram(const bool input_last_dim_divisible_by_4, const bool output_last_dim_divisible_by_4) : Program{"Expand"},
                                                                                                       input_last_dim_divisible_by_4_{input_last_dim_divisible_by_4},
                                                                                                       output_last_dim_divisible_by_4_{output_last_dim_divisible_by_4} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"data_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool input_last_dim_divisible_by_4_;
  bool output_last_dim_divisible_by_4_;
};

class Expand final : public WebGpuKernel {
 public:
  Expand(const OpKernelInfo& info) : WebGpuKernel(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
