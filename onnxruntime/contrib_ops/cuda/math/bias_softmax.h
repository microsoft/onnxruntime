// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
struct DispatchBiasSoftmaxForward {
  void operator()(
      cudaStream_t stream,
      Tensor* output,
      const Tensor* input,
      const Tensor* input_bias,
      int element_count,
      int batch_count,
      int batch_stride,
      int bias_broadcast_size_per_batch);
};

template <typename T>
struct DispatchBiasSoftMaxForwardViaDnnLibrary {
  void operator()(
      cudaStream_t stream,
      cudnnHandle_t cudaDnnHandle,
      int element_count,
      int batch_count,
      int broadcast_axis,
      int softmax_axis,
      const onnxruntime::TensorShape& X_shape,
      const onnxruntime::Tensor* X,
      const onnxruntime::TensorShape& B_shape,
      const onnxruntime::Tensor* B,
      onnxruntime::Tensor* Y);
};

class BiasSoftmax final : public onnxruntime::cuda::CudaKernel {
 public:
  BiasSoftmax(const OpKernelInfo& info) : CudaKernel{info} {
    info.GetAttrOrDefault("softmax_axis", &softmax_axis_, static_cast<int64_t>(1));
    info.GetAttrOrDefault("broadcast_axis", &broadcast_axis_, static_cast<int64_t>(1));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t softmax_axis_;
  int64_t broadcast_axis_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
