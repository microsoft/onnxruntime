// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/lrn.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T>
class LRN final : public onnxruntime::LRN<T> {
 public:
  LRN(const OpKernelInfo& info) : onnxruntime::LRN<T>(info) {}

  Status Compute(OpKernelContext* p_op_kernel_context) const override;
};

}
}