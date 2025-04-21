// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {
class MatMulProgram final : public Program<MatMulProgram> {
 public:
  MatMulProgram(const Activation& activation, bool bias, bool is_vec4, const gsl::span<int64_t>& elements_per_thread, bool is_channels_last = false) : Program{"MatMul"},
                                                                                                                                                       activation_(activation),
                                                                                                                                                       has_bias_{bias},
                                                                                                                                                       is_vec4_{is_vec4},
                                                                                                                                                       elements_per_thread_(elements_per_thread.begin(), elements_per_thread.end()),
                                                                                                                                                       is_channels_last_(is_channels_last) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"dim_a_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_inner", ProgramUniformVariableDataType::Uint32});

  static Status MakeMatMulPackedVec4Source(ShaderHelper& shader,
                                           const InlinedVector<int64_t>& elements_per_thread,
                                           uint32_t workgroup_size_x,
                                           uint32_t workgroup_size_y,
                                           const std::string& data_type,
                                           const ShaderIndicesHelper* batch_dims,
                                           bool transpose_a = false,
                                           uint32_t tile_inner = 32,
                                           bool split_k = false,
                                           uint32_t splitted_dim_inner = 32);
  static Status MakeMatMulPackedSource(ShaderHelper& shader,
                                       const InlinedVector<int64_t>& elements_per_thread,
                                       uint32_t workgroup_size_x,
                                       uint32_t workgroup_size_y,
                                       const std::string& data_type,
                                       const ShaderIndicesHelper* batch_dims,
                                       bool transpose_a = false,
                                       uint32_t tile_inner = 32,
                                       bool split_k = false,
                                       uint32_t splitted_dim_inner = 32,
                                       bool sequentially_access_by_threads = false);

 private:
  const Activation& activation_;
  const bool has_bias_;
  const bool is_vec4_;
  const InlinedVector<int64_t> elements_per_thread_;
  bool is_channels_last_ = false;
  void MatMulReadWriteFnSource(ShaderHelper& shader, const ShaderVariableHelper& a, const ShaderVariableHelper& b, const ShaderVariableHelper& output, const ShaderIndicesHelper& batch_dims, std::string apply_activation) const;
};

}  // namespace webgpu
}  // namespace onnxruntime
