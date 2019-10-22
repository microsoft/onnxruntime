// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/conv_transpose.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
class ConvTransposeWithDynamicPads : public ConvTranspose<T> {
 public:
  ConvTransposeWithDynamicPads(const OpKernelInfo& info) : ConvTranspose<T>(info) {}

  Status Compute(OpKernelContext* context) const override {
    return ConvTranspose<T>::DoConvTranspose(context, true);
  }
};
}  // namespace contrib
}  // namespace onnxruntime
