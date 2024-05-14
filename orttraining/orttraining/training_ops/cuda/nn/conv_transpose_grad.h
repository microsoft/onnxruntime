// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

#include "core/providers/cpu/nn/conv_attributes.h"
#include "orttraining/training_ops/cuda/nn/conv_shared.h"

namespace onnxruntime::cuda {

template <typename T>
class ConvTransposeGrad final : public CudaKernel {
 public:
  using CudaT = typename ToCudaType<T>::MappedType;

  ConvTransposeGrad(const OpKernelInfo& info) : CudaKernel(info), conv_attrs_(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status ComputeWeightGradient(onnxruntime::Stream* stream, const ConvArgs& args) const;
  Status ComputeInputGradient(onnxruntime::Stream* stream, const ConvArgs& args) const;
  Status ComputeBiasGradient(const ConvArgs& args) const;

  Status PrepareConvForwardArgs(const Tensor& X, const Tensor& W,
                                Tensor& Y, cudnnHandle_t cudnn_handle,
                                ConvArgs& args) const;

  Status PrepareConvBackwardFilterArgs(const Tensor& X, const Tensor& W, const Tensor& dY,
                                       Tensor* dW, Tensor* dB, cudnnHandle_t cudnn_handle,
                                       ConvArgs& args) const;

  ConvAttributes conv_attrs_;
  mutable ConvArgs args_dx_;
  mutable ConvArgs args_dw_;
};

}  // namespace onnxruntime::cuda
