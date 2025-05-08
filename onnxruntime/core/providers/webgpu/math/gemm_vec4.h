// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

class GemmVec4Program final : public Program<GemmVec4Program> {
 public:
  GemmVec4Program(bool transA, bool transB, float alpha, bool need_handle_bias, bool need_handle_matmul, int c_components, bool c_is_scalar, int output_components, bool is_vec4 = false)
      : Program{"GemmVec4"},
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

  static Status MakeMatMulPackedVec4Source(ShaderHelper& shader,
                                           const InlinedVector<int64_t>& elements_per_thread,
                                           uint32_t workgroup_size_x,
                                           uint32_t workgroup_size_y,
                                           const std::string& data_type,
                                           const ShaderIndicesHelper* batch_dims,
                                           bool transpose_a = false,
                                           bool transpose_b = false,
                                           float alpha = 1.0f,
                                           int output_components = 4,
                                           bool need_handle_matmul = true,
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
                                       bool transpose_b = false,
                                       float alpha = 1.0f,
                                       int output_components = 4,
                                       bool need_handle_matmul = true,
                                       uint32_t tile_inner = 32,
                                       bool split_k = false,
                                       uint32_t splitted_dim_inner = 32,
                                       bool sequentially_access_by_threads = false);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"num_tile_n", ProgramUniformVariableDataType::Uint32},
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"M4", ProgramUniformVariableDataType::Uint32},
      {"N4", ProgramUniformVariableDataType::Uint32},
      {"K4", ProgramUniformVariableDataType::Uint32},
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
  void MatMulReadFnSource(ShaderHelper& shader, const ShaderVariableHelper& a,
                          const ShaderVariableHelper& b,
                          const ShaderVariableHelper& output,
                          const ShaderIndicesHelper& batch_dims) const;
  void MatMulWriteFnSource(ShaderHelper& shader, const ShaderVariableHelper& output) const;
};

Status ApplyGemmVec4(const Tensor* a,
                     const Tensor* b,
                     const Tensor* c,
                     bool transA,
                     bool transB,
                     float alpha,
                     float beta,
                     ComputeContext& context,
                     Tensor* y);

}  // namespace webgpu
}  // namespace onnxruntime
