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
                                          {"logical_dispatch_z", ProgramUniformVariableDataType::Uint32});

  bool NeedSplitK() const;

 private:
  const Activation activation_;
  const bool has_bias_;
  const bool is_vec4_;
  const InlinedVector<int64_t> elements_per_thread_;
  bool is_channels_last_ = false;
  uint32_t split_dim_inner_ = 1;
};

// The program to initialize the output with 0 or bias before doing MatMul with Split-K. In Split-K,
// we set the output values with `atomicLoad` and `atomicCompareExchangeWeak` instead of a direct
// assignment (see the function `HandleMatMulWithSplitK()` in `gemm_utils.cc`), so we must initialize
// the output with 0 or bias first to make sure `atomicLoad` won't return garbage data.
class MatMulFillBiasOrZeroBeforeSplitKProgram final : public Program<MatMulFillBiasOrZeroBeforeSplitKProgram> {
 public:
  MatMulFillBiasOrZeroBeforeSplitKProgram(bool is_gemm, bool has_bias, uint32_t output_components, bool bias_is_scalar)
      : Program{"MatMul_Fill_Bias_Or_Zero_Before_Split_K"},
        is_gemm_(is_gemm),
        has_bias_(has_bias),
        output_components_(output_components),
        bias_is_scalar_(bias_is_scalar) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"dim_a_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
                                          {"beta", ProgramUniformVariableDataType::Float32});

 private:
  bool is_gemm_ = false;
  bool has_bias_ = false;
  uint32_t output_components_ = 0;
  bool bias_is_scalar_ = false;
};

}  // namespace webgpu
}  // namespace onnxruntime
