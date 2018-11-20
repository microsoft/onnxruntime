// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv.h"

namespace onnxruntime {
namespace mkl_dnn {
template <typename T>
class Conv final : public onnxruntime::Conv<T> {
 public:
  Conv(const OpKernelInfo& info) : onnxruntime::Conv<T>(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
};
}  // namespace mkl_dnn
}  // namespace onnxruntime
