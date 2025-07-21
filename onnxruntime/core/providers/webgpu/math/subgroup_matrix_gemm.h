// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

class SubgroupMatrixGemmProgram final : public Program<SubgroupMatrixGemmProgram> {
 public:
  SubgroupMatrixGemmProgram(bool transA, bool transB, float alpha, bool need_handle_bias)
      : Program{"SubgroupMatrixGemm"},
        transA_(transA),
        transB_(transB),
        alpha_(alpha),
        need_handle_bias_(need_handle_bias) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32},
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32});

  constexpr static uint32_t SUBGROUP_MATRIX_WORKGROUP_SIZE_X = 128;
  constexpr static uint32_t SUBGROUP_MATRIX_WORKGROUP_SIZE_Y = 1;
  constexpr static uint32_t SUBGROUP_MATRIX_WORKGROUP_SIZE_Z = 1;

 private:
  bool transA_;
  bool transB_;
  float alpha_;
  bool need_handle_bias_;
};

bool CanApplySubgroupMatrixGemm(ComputeContext& context, uint32_t K, uint32_t N);

Status ApplySubgroupMatrixGemm(const Tensor* a,
                               const Tensor* b,
                               const Tensor* c,
                               bool transA,
                               bool transB,
                               float alpha,
                               float beta,
                               ComputeContext& context);

}  // namespace webgpu
}  // namespace onnxruntime

#endif
