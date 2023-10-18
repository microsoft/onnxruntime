// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/nn/conv_transpose.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class ConvTransposeWithDynamicPads : public ::onnxruntime::cuda::ConvTranspose<T, false> {
 public:
  ConvTransposeWithDynamicPads(const OpKernelInfo& info) : ::onnxruntime::cuda::ConvTranspose<T, false>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    return ::onnxruntime::cuda::ConvTranspose<T, false>::DoConvTranspose(context, true);
  }
};
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
