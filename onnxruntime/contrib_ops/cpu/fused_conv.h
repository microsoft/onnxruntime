// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/conv_impl.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class FusedConv : public Conv<T> {
 public:
  FusedConv(const OpKernelInfo& info) : Conv<T>(info) {
    Conv<T>::activation_ = info.GetAttrOrDefault<std::string>("activation", "");
    Conv<T>::alpha_ = info.GetAttrOrDefault("alpha", 0.01f);
  }

  Status Compute(OpKernelContext* context) const override {
    return Conv<T>::Compute(context);
  }
};
}  // namespace contrib
}  // namespace onnxruntime
