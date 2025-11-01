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
                                          {"dim_inner", ProgramUniformVariableDataType::Uint32});

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
class MatMulFillBiasBeforeSplitKProgram final : public Program<MatMulFillBiasBeforeSplitKProgram> {
 public:
  explicit MatMulFillBiasBeforeSplitKProgram(bool has_bias)
      : Program{"MatMul_Fill_Bias_Before_Split_K"},
        has_bias_(has_bias) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"dim_a_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_b_outer", ProgramUniformVariableDataType::Uint32});

  constexpr static uint32_t WORKGROUP_SIZE_X = 8;
  constexpr static uint32_t WORKGROUP_SIZE_Y = 8;
  constexpr static uint32_t ELEMENTS_PER_THREAD = 8;

 private:
  bool has_bias_ = false;
};

}  // namespace webgpu
}  // namespace onnxruntime
