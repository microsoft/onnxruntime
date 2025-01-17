// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/training_ops/cuda/nn/conv_shared.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class ConvGrad final : public CudaKernel {
 public:
  using CudaT = typename ToCudaType<T>::MappedType;

  ConvGrad(const OpKernelInfo& info) : CudaKernel(info), conv_attrs_(info) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
    ORT_THROW("ConvGrad CUDA kernel is not yet tested on __CUDA_ARCH__ lower than 700");
#endif
    auto pads_size = conv_attrs_.pads.size();
    ORT_ENFORCE(pads_size % 2 == 0);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  Status PrepareArgs(const Tensor& x, const Tensor& dY, const Tensor& w, Tensor* dB, Tensor* dX, Tensor* dW, cudnnHandle_t cudnn_handle) const;
  mutable ConvArgs args_;
  ConvAttributes conv_attrs_;

 private:
  Status ComputeWeightGradient(onnxruntime::Stream* stream) const;
  Status ComputeInputGradient(onnxruntime::Stream* stream) const;
  Status ComputeBiasGradient() const;
};

}  // namespace cuda
}  // namespace onnxruntime
