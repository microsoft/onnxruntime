// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"

namespace onnxruntime {
namespace cuda {

// Trait class for inplace broadcast
class ShouldBroadcastInplace {
};

template <typename BroadcastTrait>
class BinaryElementwiseInplace : public CudaKernel {
 public:
  using CudaKernel::CudaKernel;

  void SetInOutIndexBeforePrepare(int inout_index, int input_index) const {
    inout_index_ = inout_index;
    input_index_ = input_index;
  }

  Status Prepare(OpKernelContext* context, int device_id, BinaryElementwisePreparation* p) const;

 private:
  mutable int inout_index_;
  mutable int input_index_;
};

class SGDOptimizer final : public BinaryElementwiseInplace<ShouldBroadcastInplace> {
 public:
  SGDOptimizer(const OpKernelInfo& info) : BinaryElementwiseInplace(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
void AdamOptimizerImpl(
    const T* eta,
    int64_t* update_count,
    const T* weights,
    const T* grads,
    const T* moment_1,
    const T* moment_2,
    float alpha,
    float beta,
    float lambda,
    float epsilon,
    T* weight_out,
    T* moment_1_out,
    T* moment_2_out,
    size_t count);

class AdamOptimizer final : public CudaKernel {
 public:
  AdamOptimizer(const OpKernelInfo& info): CudaKernel(info) {
    info.GetAttrOrDefault("alpha", &alpha_, 0.9f);
    info.GetAttrOrDefault("beta", &beta_, 0.999f);
    info.GetAttrOrDefault("lambda", &lambda_, 0.0f);
    info.GetAttrOrDefault("epsilon", &epsilon_, 1e-6f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
   float alpha_;
   float beta_;
   float lambda_;
   float epsilon_;
};

}  // namespace cuda
}  // namespace onnxruntime
