// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
class MatMul final : public CudaKernel {
  using Base = CudaKernel;

 public:
  MatMul(const OpKernelInfo& info)
      : CudaKernel(info),
        alpha_{info.GetAttrOrDefault<float>("alpha", 1.0f)},
        trans_A_{info.GetAttrOrDefault<int64_t>("transA", 0) != 0},
        trans_B_{info.GetAttrOrDefault<int64_t>("transB", 0) != 0},
        trans_batch_a_{info.GetAttrOrDefault<int64_t>("transBatchA", 0) != 0},
        trans_batch_b_{info.GetAttrOrDefault<int64_t>("transBatchB", 0) != 0},
        use_fp8_("1" == info.GetConfigOptions().GetConfigEntry(kOrtSessionOptionsGemmCudaFloat8E4M3FN)),
        right_X_fp8_(nullptr)
        {
          if (use_fp8_) {
            // TODO should we just initialize the epilogue to CUBLASLT_EPILOGUE_DEFAULT?
            // Or should "activation" be added as an attribute in contrib_defs.cc?
            std::string activation = info.GetAttrOrDefault<std::string>("activation", "NONE");
            if (activation == "NONE") {
              epilogue_ = CUBLASLT_EPILOGUE_DEFAULT;
            } else if (activation == "RELU") {
              epilogue_ = CUBLASLT_EPILOGUE_RELU;
            } else if (activation == "GELU") {
              epilogue_ = CUBLASLT_EPILOGUE_GELU;
            } else {
              ORT_THROW("Unexpected value for activation: '", activation, "'.");
            }
          }
        }

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc, bool& is_packed,
                 PrePackedWeights* prepacked_weights) override;
  Status ComputeInternal(OpKernelContext* context) const override;
  Status ComputeDefault(OpKernelContext* context, MatMulComputeHelper& helper) const;

 private:
  const float alpha_;
  const bool trans_A_;
  const bool trans_B_;
  const bool trans_batch_a_;
  const bool trans_batch_b_;
  const bool use_fp8_;
  cublasLtEpilogue_t epilogue_;
  mutable IAllocatorUniquePtr<void> right_X_fp8_; // mutable because ComputeDefault has to be const

  Status ComputeDefaultImpl(OpKernelContext* context, MatMulComputeHelper& helper) const;
};

template <typename T>
Status FuncMatMul(
    // Use OpKernel and do a pointer cast to unify functional calls with other eps.
    // TODO: remove CudaKernel and OpKernelContext.
    const CudaKernel* cuda_kernel,
    // Do NOT use ctx to access inputs and outputs.
    // Inputs and outputs are passed in as function arguments.
    OpKernelContext* ctx,
    const Tensor* A,
    const Tensor* B,
    float alpha,
    bool trans_A,
    bool trans_B,
    bool trans_batch_A,
    bool trans_batch_B,
    Tensor* Y);

}  // namespace cuda
}  // namespace onnxruntime
