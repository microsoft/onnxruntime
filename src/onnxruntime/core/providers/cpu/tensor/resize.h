// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cpu/tensor/upsample.h"

namespace onnxruntime {
template <typename T>
class Resize : public Upsample<T> {
 public:
  Resize(const OpKernelInfo& info) : Upsample<T>(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    return Upsample<T>::Compute(context);
  }
};

}  // namespace onnxruntime
