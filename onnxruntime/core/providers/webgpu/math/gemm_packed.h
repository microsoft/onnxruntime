// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

class GemmProgram final : public Program<GemmProgram> {
 public:
  GemmProgram(bool transA, bool transB, float alpha, bool need_handle_bias, bool need_handle_matmul, int c_components, bool c_is_scalar, int output_components, bool is_vec4 = false)
      : Program{"Gemm"},
        transA_{transA},
        transB_{transB},
        alpha_{alpha},
        need_handle_bias_{need_handle_bias},
        need_handle_matmul_{need_handle_matmul},
        c_components_(c_components),
        c_is_scalar_(c_is_scalar),
        output_components_(output_components),
        is_vec4_(is_vec4) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32},
      {"dim_a_outer", ProgramUniformVariableDataType::Uint32},
      {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
      {"dim_inner", ProgramUniformVariableDataType::Uint32});

  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_X = 8;
  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_Y = 8;
  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_Z = 1;

 private:
  bool transA_;
  bool transB_;
  float alpha_;
  bool need_handle_bias_;
  bool need_handle_matmul_;
  int c_components_;
  bool c_is_scalar_ = false;
  int output_components_;
  bool is_vec4_ = false;
};

Status ApplyGemmPacked(const Tensor* a,
                       const Tensor* b,
                       const Tensor* c,
                       bool transA,
                       bool transB,
                       float alpha,
                       float beta,
                       ComputeContext& context);

}  // namespace webgpu
}  // namespace onnxruntime
