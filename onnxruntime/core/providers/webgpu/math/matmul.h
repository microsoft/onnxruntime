// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/math/matmul_compute_dispatcher.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/math/matmul_packed.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

MatMulFillBiasOrZeroBeforeSplitKProgram CreateMatMulFillBiasOrZeroBeforeSplitKProgram(
    const Tensor* bias,
    Tensor* output,
    bool is_gemm,
    float beta,
    uint32_t output_components,
    const TensorShape& output_shape,
    uint32_t batch_size = 1);

class MatMul final : public WebGpuKernel {
 public:
  MatMul(const OpKernelInfo& info) : WebGpuKernel{info} {
    // Whether the B (weight) input is a constant initializer. The subgroup-matrix
    // opt impl uses this to decide it can safely pad B once and cache the result
    // (odd-N handling); a non-constant B changes per run and must not be cached.
    const Tensor* b = nullptr;
    b_is_constant_ = info.TryGetConstantInput(1, &b);
  }

  Status ComputeInternal(ComputeContext& context) const override;

  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_X = 8;
  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_Y = 8;
  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_Z = 1;

 private:
  mutable MatMulComputeDispatcher compute_dispatcher_;
  bool b_is_constant_ = false;
};

#define WEBGPU_MAT_MUL_NAIVE_PROGRAM_CONFIG(F) \
  F(ShaderActivation, activation_)             \
  F(size_t, output_rank_)                      \
  F(int64_t, output_number_)                   \
  F(bool, has_bias_)                           \
  F(bool, is_channels_last_)                   \
  F(bool, is_zero_k_)

struct MatMulNaiveProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_MAT_MUL_NAIVE_PROGRAM_CONFIG);
  static Config MakeConfig(const Activation& activation, const size_t output_rank, int64_t output_number,
                           bool has_bias, bool is_channels_last = false, bool is_zero_k = false) {
    Config result{};
    result.activation_ = MakeShaderActivation(activation);
    result.output_rank_ = size_t(output_rank);
    result.output_number_ = int64_t(output_number);
    result.has_bias_ = bool{has_bias};
    result.is_channels_last_ = bool(is_channels_last);
    result.is_zero_k_ = bool(is_zero_k);

    return result;
  }
  static constexpr std::string_view name = "MatMulNaive";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_MAT_MUL_NAIVE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                \
  U(Uint32, M)                                          \
  U(Uint32, N)                                          \
  U(Uint32, K)                                          \
  U(Float32, activation_param_0)                        \
  U(Float32, activation_param_1)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_MAT_MUL_NAIVE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_MAT_MUL_NAIVE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_MAT_MUL_NAIVE_PROGRAM_CONFIG

using MatMulNaiveProgram = ConfiguredProgram<MatMulNaiveProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
