// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class ConcatProgram final : public Program<ConcatProgram> {
 public:
  ConcatProgram(size_t axis) : Program{"Concat"}, axis_{axis} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size_in_concat_axis", ProgramUniformVariableDataType::Uint32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  size_t axis_;
};

class Concat final : public WebGpuKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : WebGpuKernel(info), ConcatBase(info) {
  }

  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
