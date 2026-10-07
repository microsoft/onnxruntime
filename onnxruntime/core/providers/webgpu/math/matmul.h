// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/math/matmul_packed.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

class MatMulOptImpl {
 public:
  virtual ~MatMulOptImpl() = default;

  virtual Status Compute(ComputeContext& context,
                         const std::vector<const Tensor*>& inputs,
                         Tensor* output,
                         const Activation& activation,
                         bool is_channels_last,
                         bool b_is_constant,
                         /*out*/ bool& handled) = 0;
};

class MatMulOptImplCache {
 public:
  MatMulOptImplCache() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulOptImplCache);

  MatMulOptImpl* GetOrCreate(const ComputeContextBase& context);

 private:
  std::once_flag subgroup_impl_init_flag_;
  std::unique_ptr<MatMulOptImpl> subgroup_impl_;
};

Status ComputeMatMul(ComputeContext* context, const Activation& activation, std::vector<const Tensor*>& inputs, Tensor* output,
                     bool is_channels_last, MatMulOptImplCache& cache,
                     bool b_is_constant = false);

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
  mutable MatMulOptImplCache compute_cache_;
  bool b_is_constant_ = false;
};

#define WEBGPU_MAT_MUL_NAIVE_PROGRAM_CONFIG(F) \
  F(ShaderActivation, activation_)             \
  F(size_t, output_rank_)                      \
  F(int64_t, output_number_)                   \
  F(bool, has_bias_)                           \
  F(bool, is_channels_last_)

struct MatMulNaiveProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MAT_MUL_NAIVE_PROGRAM_CONFIG);
    Config(const Activation& activation, const size_t output_rank, int64_t output_number, bool has_bias,
           bool is_channels_last = false)
        : activation_(activation),
          output_rank_(output_rank),
          output_number_(output_number),
          has_bias_{has_bias},
          is_channels_last_(is_channels_last) {}
  };
  static constexpr std::string_view name = "MatMulNaive";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);
};
#undef WEBGPU_MAT_MUL_NAIVE_PROGRAM_CONFIG

using MatMulNaiveProgram = ConfiguredProgram<MatMulNaiveProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
