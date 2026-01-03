// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

class GemmSubgroupProgram final : public Program<GemmSubgroupProgram> {
 public:
  GemmSubgroupProgram(bool transA, bool transB, float alpha, bool need_handle_bias, bool need_handle_matmul,
                      int c_components, bool c_is_scalar, bool is_vec4,
                      const gsl::span<int64_t>& elements_per_thread)
      : Program{"GemmSubgroup"},
        transA_{transA},
        transB_{transB},
        alpha_{alpha},
        need_handle_bias_{need_handle_bias},
        need_handle_matmul_{need_handle_matmul},
        c_components_(c_components),
        c_is_scalar_(c_is_scalar),
        is_vec4_(is_vec4),
        elements_per_thread_(elements_per_thread.begin(), elements_per_thread.end()) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32},
      {"dim_a_outer", ProgramUniformVariableDataType::Uint32},
      {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
      {"dim_inner", ProgramUniformVariableDataType::Uint32});

 private:
  bool transA_;
  bool transB_;
  float alpha_;
  bool need_handle_bias_;
  bool need_handle_matmul_;
  int c_components_;
  bool c_is_scalar_ = false;
  bool is_vec4_ = false;
  const InlinedVector<int64_t> elements_per_thread_;
};

bool CanApplyGemmIntel(const ComputeContext& context, int64_t M, int64_t N, int64_t K, bool transA, bool transB);

Status ApplyGemmIntel(const Tensor* a,
                      const Tensor* b,
                      const Tensor* c,
                      bool transA,
                      bool transB,
                      float alpha,
                      float beta,
                      ComputeContext& context);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
