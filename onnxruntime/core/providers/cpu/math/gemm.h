// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include "core/util/math.h"
#include "core/providers/cpu/activation/activations.h"

namespace onnxruntime {

template <typename T>
class Gemm : public OpKernel {
 public:
  Gemm(const OpKernelInfo& info) : OpKernel(info) {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed,
                 /*in_out*/ PackedWeight& cached_prepacked_tensor,
                 /*out*/ bool& read_from_cache,
                 AllocatorPtr alloc_for_caching) override;

  static void ComputeGemm(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                          int64_t M, int64_t N, int64_t K,
                          float alpha,
                          const T* a_data, const T* b_data,
                          float beta,
                          const T* c_data, const TensorShape* c_shape,
                          T* y_data,
                          concurrency::ThreadPool* thread_pool);

 private:
  CBLAS_TRANSPOSE trans_A_;
  CBLAS_TRANSPOSE trans_B_;
  float alpha_;
  float beta_;

 protected:
  TensorShape b_shape_;
  BufferUniquePtr packed_b_;

  // For fused gemm + activation
  std::unique_ptr<functors::ElementWiseRangedTransform<T>> activation_;

  void ComputeActivation(T* y_data, size_t y_size, concurrency::ThreadPool* thread_pool) const;
};

}  // namespace onnxruntime
