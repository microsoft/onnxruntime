// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "softmax.h"
#include "core/providers/cuda/reduction/reduction_ops.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void SoftMaxCrossEntropyImpl(
    const T* prob,
    const T* label,
    T* output_data,
    size_t count);

template <typename T>
void SoftMaxCrossEntropyGradImpl(
    const T* dY,
    const T* prob,
    const T* label,
    T* output_data,
    size_t count);

template <typename T, typename Tin>
void SparseSoftmaxCrossEntropyImpl(
    const T* prob,
    const Tin* label,
    const T* weight,
    T* output_data,
    size_t count,
    size_t label_depth);

template <typename T, typename Tin>
void SparseSoftmaxCrossEntropyGradImpl(
    const T* dY,
    const T* prob,
    const Tin* label,
    const T* weight,
    T* output_data,
    size_t count,
    size_t label_depth);

template <typename T>
class SoftmaxCrossEntropy final : public ReduceKernel<true> {
 public:
  SoftmaxCrossEntropy(const OpKernelInfo& info) : ReduceKernel<true>(info, std::make_unique<int64_t>(0)) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class SoftmaxCrossEntropyGrad final : public CudaKernel {
 public:
  SoftmaxCrossEntropyGrad(const OpKernelInfo& info) : CudaKernel{info} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, typename Tin>
class SparseSoftmaxCrossEntropy final : public ReduceKernel<true> {
 public:
  SparseSoftmaxCrossEntropy(const OpKernelInfo& info) : ReduceKernel<true>(info, std::make_unique<int64_t>(0)) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, typename Tin>
class SparseSoftmaxCrossEntropyGrad final : public CudaKernel {
 public:
  SparseSoftmaxCrossEntropyGrad(const OpKernelInfo& info) : CudaKernel{info} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
