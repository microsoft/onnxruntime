// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/math/matmul_utils.h"

namespace onnxruntime {
namespace webgpu {
class MatMulProgram final : public Program<MatMulProgram> {
 public:
  MatMulProgram(bool bias, bool is_vec4, const gsl::span<int64_t>& elements_per_thread) : Program{"MatMul"},
                                                                                          has_bias_{bias},
                                                                                          is_vec4_{is_vec4},
                                                                                          elements_per_thread_(elements_per_thread.begin(), elements_per_thread.end()) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"dim_a_outer", ProgramUniformVariableDataType::Int32},
                                          {"dim_b_outer", ProgramUniformVariableDataType::Int32},
                                          {"dim_inner", ProgramUniformVariableDataType::Int32});

  static Status MakeMatMulPackedVec4Source(ShaderHelper& shader,
                                           const ShaderIndicesHelper& batch_dims,
                                           const InlinedVector<int64_t>& elements_per_thread,
                                           uint32_t workgroup_size_x,
                                           uint32_t workgroup_size_y);
  static Status MakeMatMulPackedSource(ShaderHelper& shader,
                                       const ShaderIndicesHelper& batch_dims,
                                       const InlinedVector<int64_t>& elements_per_thread,
                                       uint32_t workgroup_size_x,
                                       uint32_t workgroup_size_y);

 private:
  const bool has_bias_;
  const bool is_vec4_;
  const InlinedVector<int64_t> elements_per_thread_;

  void MatMulReadWriteFnSource(ShaderHelper& shader, const ShaderVariableHelper& a, const ShaderVariableHelper& b, const ShaderVariableHelper& output, const ShaderIndicesHelper& batch_dims) const;
};

}  // namespace webgpu
}  // namespace onnxruntime
