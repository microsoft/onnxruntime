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
  MatMulProgram(const Activation& activation, bool bias, bool is_vec4, const gsl::span<int64_t>& elements_per_thread, bool is_channels_last = false, uint32_t split_dim_inner = 1) : Program{"MatMul"},
                                                                                                                                                                                     activation_(activation),
                                                                                                                                                                                     has_bias_{bias},
                                                                                                                                                                                     is_vec4_{is_vec4},
                                                                                                                                                                                     elements_per_thread_(elements_per_thread.begin(), elements_per_thread.end()),
                                                                                                                                                                                     is_channels_last_(is_channels_last),
                                                                                                                                                                                     split_dim_inner_(split_dim_inner) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"dim_a_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_inner", ProgramUniformVariableDataType::Uint32},
                                          {"logical_dispatch_x", ProgramUniformVariableDataType::Uint32},
                                          {"logical_dispatch_y", ProgramUniformVariableDataType::Uint32},
                                          {"logical_dispatch_z", ProgramUniformVariableDataType::Uint32},
                                          {"splits_per_batch", ProgramUniformVariableDataType::Uint32},
                                          WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);

  bool NeedSplitK() const;

 private:
  const Activation activation_;
  const bool has_bias_;
  const bool is_vec4_;
  const InlinedVector<int64_t> elements_per_thread_;
  bool is_channels_last_ = false;
  uint32_t split_dim_inner_ = 1;
};

// Pass 2 of the deterministic two-pass Split-K reduction.
//
// Pass 1 leaves each split's partial sum in its own slot of a scratch buffer laid out split-major:
//
//   partials[split_index * out_elems + output_id],  out_elems = batch_size * dim_a_outer * dim_b_outer_vec
//
// This runs one invocation per output element, sums the `splits` slots in a fixed index order,
// applies bias (or `beta * C` for GEMM), and writes the output once. Fixing the order by index
// rather than by the order pass-1 workgroups retire is what makes the result reproducible.
//
// It subsumes MatMul_Fill_Bias_Or_Zero_Before_Split_K, which the atomic path needed to seed the
// output before `atomicLoad`; nothing is read back now, so the path still costs two dispatches.
//
// `splits` is a uniform rather than a shader constant so one pipeline serves every split count.
class MatMulSplitKReduceProgram final : public Program<MatMulSplitKReduceProgram> {
 public:
  MatMulSplitKReduceProgram(bool is_gemm, bool has_bias, uint32_t output_components, bool bias_is_scalar,
                            const Activation& activation)
      : Program{"MatMul_Split_K_Reduce"},
        is_gemm_(is_gemm),
        has_bias_(has_bias),
        output_components_(output_components),
        bias_is_scalar_(bias_is_scalar),
        activation_(activation) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"dim_a_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
                                          {"beta", ProgramUniformVariableDataType::Float32},
                                          {"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"splits", ProgramUniformVariableDataType::Uint32},
                                          WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);

 private:
  bool is_gemm_ = false;
  bool has_bias_ = false;
  uint32_t output_components_ = 0;
  bool bias_is_scalar_ = false;
  Activation activation_;
};

}  // namespace webgpu
}  // namespace onnxruntime
