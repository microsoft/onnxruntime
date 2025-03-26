// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/math/matmul_packed.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

class MatMul final : public WebGpuKernel {
 public:
  MatMul(const OpKernelInfo& info) : WebGpuKernel{info} {}

  Status ComputeInternal(ComputeContext& context) const override;

  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_X = 8;
  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_Y = 8;
  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_Z = 1;
};

class MatMulNaiveProgram final : public Program<MatMulNaiveProgram> {
 public:
  MatMulNaiveProgram(const size_t output_rank, int64_t output_number, bool has_bias)
      : Program{"MatMulNaive"}, output_rank_(output_rank), output_number_(output_number), has_bias_{has_bias} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32});

 private:
  const size_t output_rank_;
  const int64_t output_number_;
  const bool has_bias_;
};

}  // namespace webgpu
}  // namespace onnxruntime
