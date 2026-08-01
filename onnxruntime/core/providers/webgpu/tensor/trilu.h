// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class TriluProgram final : public Program<TriluProgram> {
 public:
  explicit TriluProgram(bool upper) : Program{"Trilu"}, upper_{upper} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"matrix_h", ProgramUniformVariableDataType::Uint32},
                                          {"matrix_w", ProgramUniformVariableDataType::Uint32},
                                          {"k", ProgramUniformVariableDataType::Int32});

 private:
  bool upper_;
};

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
