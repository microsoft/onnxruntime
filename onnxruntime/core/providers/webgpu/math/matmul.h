// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/math/matmul_naive.h"
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
                         const TensorShape& a_shape,
                         const TensorShape& b_shape,
                         const TensorShape& output_shape,
                         Tensor* output,
                         const Activation& activation,
                         bool is_channels_last,
                         bool b_is_constant,
                         bool has_persistent_cache,
                         /*out*/ bool& handled) = 0;
};

class MatMulComputeCache {
 public:
  MatMulComputeCache() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulComputeCache);

  MatMulOptImpl* GetOrCreateSubgroupMatrixImpl(const ComputeContextBase& context);

 private:
  std::once_flag subgroup_impl_init_flag_;
  std::unique_ptr<MatMulOptImpl> subgroup_impl_;
};

Status ComputeMatMul(ComputeContext* context, const Activation& activation, std::vector<const Tensor*>& inputs, Tensor* output, bool is_channels_last = true,
                     MatMulComputeCache* cache = nullptr,
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
  mutable MatMulComputeCache compute_cache_;
  bool b_is_constant_ = false;
};

class MatMulNaiveProgram final : public Program<MatMulNaiveProgram> {
 public:
  MatMulNaiveProgram(const Activation& activation, const size_t output_rank, int64_t output_number, bool has_bias, bool is_channels_last = false)
      : Program{"MatMulNaive"}, activation_(activation), output_rank_(output_rank), output_number_(output_number), has_bias_{has_bias}, is_channels_last_(is_channels_last) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);

 private:
  const Activation activation_;
  const size_t output_rank_;
  const int64_t output_number_;
  const bool has_bias_;
  const bool is_channels_last_;
};

}  // namespace webgpu
}  // namespace onnxruntime
