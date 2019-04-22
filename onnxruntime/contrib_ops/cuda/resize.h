// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/tensor/upsample.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class Resize : public ::onnxruntime::cuda::Upsample<T> {
 public:
  Resize(OpKernelInfo info) : Upsample(info) {
    UpsampleBase::is_resize = true;
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    return Upsample::ComputeInternal(context);
  }
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
