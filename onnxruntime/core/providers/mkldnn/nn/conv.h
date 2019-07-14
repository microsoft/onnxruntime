// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_base.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T>
class Conv final : public OpKernel, public onnxruntime::ConvBase {
 public:
  explicit Conv(const OpKernelInfo& info) : OpKernel(info), onnxruntime::ConvBase(info) {
    provider_ = (const_cast<MKLDNNExecutionProvider*>(
        dynamic_cast<const MKLDNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  MKLDNNExecutionProvider* provider_;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime
