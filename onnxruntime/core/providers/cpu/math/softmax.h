// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl-lite.hpp"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/softmax_shared.h"

namespace onnxruntime {
template <typename T>
class Softmax final : public OpKernel {
 public:
  Softmax(const OpKernelInfo& info) : OpKernel{info} {
    const auto& node = info.node();
    opset_ = node.SinceVersion();

    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    } else {
      if (opset_ < 13) {
        axis_ = 1;  // opset-12 and below, the default axis value is 1
      } else {
        axis_ = -1;  // opset-13, the default axis value is -1
      }
    }

    log_softmax_ = info.GetKernelDef().OpName() == "LogSoftmax";
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  Status ComputeImpl(const Tensor& input, Tensor& output, size_t axis,
                     concurrency::ThreadPool* thread_pool) const;

  Status ComputeImplOpset13(const Tensor& input, Tensor& output, size_t axis,
                            concurrency::ThreadPool* thread_pool, OpKernelContext* ctx) const;

  int axis_;
  int opset_;
  bool log_softmax_;
};

}  // namespace onnxruntime
