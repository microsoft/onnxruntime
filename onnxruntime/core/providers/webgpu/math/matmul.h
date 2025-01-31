// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace webgpu {

class MatMul final : public WebGpuKernel {
 public:
  MatMul(const OpKernelInfo& info) : WebGpuKernel{info} {}

  Status ComputeInternal(ComputeContext& context) const override;
};

class MatMulProgram final : public Program<MatMulProgram> {
 public:
  MatMulProgram() : Program{"MatMul"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  // uniform variables


};

class MatMulNativeProgram final: public Program<MatMulNativeProgram> {
 public:
  MatMulNativeProgram(const int64_t output_size, const gsl::span<const int64_t>& outer_dims)
      : Program{"MatMulNative"}, output_size_(output_Size), outer_dims_(outer_dims.begin(), outer_dims.end()) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  // uniform variables output_size, M,N, K
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                         {"M", ProgramUniformVariableDataType::Uint32},
                                         {"N", ProgramUniformVariableDataType::Uint32},
                                         {"K", ProgramUniformVariableDataType::Uint32});


  private:
    const int64_t output_size_;
    const TensorShapeVector outer_dims_;
};

}  // namespace webgpu
} // namespace onnxruntime
