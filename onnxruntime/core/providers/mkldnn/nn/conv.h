// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv.h"
#include "../mkldnn_execution_provider.h"

namespace onnxruntime {
namespace mkl_dnn {
template <typename T>
class Conv final : public onnxruntime::Conv<T> {
 public:
  explicit Conv(const OpKernelInfo& info) : onnxruntime::Conv<T>(info) {
      provider_ = (const_cast<MKLDNNExecutionProvider*>(
        dynamic_cast<const MKLDNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
   MKLDNNExecutionProvider * provider_;
   mutable std::mutex mutex_;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime
