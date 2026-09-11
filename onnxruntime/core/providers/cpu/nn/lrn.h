// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename T>
class LRN : public OpKernel {
 public:
  LRN(const OpKernelInfo& info) : OpKernel(info) {
    int64_t size;
    ORT_THROW_IF_ERROR(info.GetAttr<int64_t>("size", &size));
    size_ = narrow<ptrdiff_t>(size);
    ORT_ENFORCE(size_ > 0);
    ORT_ENFORCE(size_ % 2 == 1);
    ORT_THROW_IF_ERROR(info.GetAttr<float>("alpha", &alpha_));
    ORT_ENFORCE(alpha_ > 0.0f);
    ORT_THROW_IF_ERROR(info.GetAttr<float>("beta", &beta_));
    ORT_ENFORCE(beta_ > 0.0f);
    ORT_THROW_IF_ERROR(info.GetAttr<float>("bias", &bias_));
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 protected:
  float alpha_;
  float beta_;
  float bias_;
  ptrdiff_t size_;
};
}  // namespace onnxruntime
