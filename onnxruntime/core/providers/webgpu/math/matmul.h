// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/math/matmul_packed.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

Status ComputeMatMul(ComputeContext* context, const Activation& activation, std::vector<const Tensor*>& inputs, Tensor* output, bool is_channels_last = true,
                     const TensorShape& input_a_reshape = TensorShape(),
                     const TensorShape& input_b_reshape = TensorShape());

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
  // Abstract base class for alternative optimized MatMul implementations.
  // Implementations can provide optimized computation paths by deriving from this class.
  class MatMulOptImpl {
   public:
    explicit MatMulOptImpl(const MatMul& parent) : parent_(parent) {}
    virtual ~MatMulOptImpl() = default;

    // Called during Compute phase to execute implementation-specific computation.
    // @param context       The WebGPU compute context.
    // @param handled       Output parameter. Set to true if this implementation handled the computation.
    // @return Status::OK() on success, or an error status on failure.
    virtual Status Compute(ComputeContext& context,
                           /*out*/ bool& handled) = 0;

   protected:
    const MatMul& parent_;
  };

  MatMul(const OpKernelInfo& info) : WebGpuKernel{info} {}

  Status ComputeInternal(ComputeContext& context) const override;

  Status PrePackInternal(ComputeContextBase& context,
                         const Tensor& tensor,
                         int input_idx,
                         AllocatorPtr alloc,
                         /*out*/ bool& is_packed) override;

  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_X = 8;
  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_Y = 8;
  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_Z = 1;

 private:
  // Alternative optimized implementation (created during PrePack if applicable)
  mutable std::unique_ptr<MatMulOptImpl> impl_;
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
                                          {"K", ProgramUniformVariableDataType::Uint32});

 private:
  const Activation activation_;
  const size_t output_rank_;
  const int64_t output_number_;
  const bool has_bias_;
  const bool is_channels_last_;
};

}  // namespace webgpu
}  // namespace onnxruntime
