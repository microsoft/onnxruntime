// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/cpu/nn/deform_conv_attributes.h"

namespace onnxruntime {
namespace webgpu {

class DeformConvProgram final : public Program<DeformConvProgram> {
 public:
  DeformConvProgram(bool has_bias, bool has_mask)
      : Program("DeformConv"), has_bias_(has_bias), has_mask_(has_mask) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"kernel_shape", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32},
      {"x_spatial", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"groups", ProgramUniformVariableDataType::Uint32},
      {"c_per_offset_group", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
  bool has_mask_;
};

class DeformConv final : public WebGpuKernel {
 public:
  DeformConv(const OpKernelInfo& info) : WebGpuKernel(info), attrs_(info) {}

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  DeformConvAttributes attrs_;
};

}  // namespace webgpu
}  // namespace onnxruntime
