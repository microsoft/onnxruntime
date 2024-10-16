// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include <core/common/safeint.h>

namespace onnxruntime {
namespace contrib {

using onnxruntime::OpKernelContext;
using onnxruntime::OpKernelInfo;

template <typename T>
Status LaunchUnfoldTensor(
    const T* input,
    T* output,
    int64_t leading_dims_size,
    int64_t unfold_dim_size,
    int64_t tailing_dims_size,
    int64_t unfold_size,
    int64_t step_size);

class UnfoldTensor final : public OpKernel {
 public:
  UnfoldTensor(const OpKernelInfo& info) : OpKernel(info) {
    dim_ = SafeInt<int>(info.GetAttrOrDefault<int64_t>("dim", -1LL));
    step_ = SafeInt<int>(info.GetAttrOrDefault<int64_t>("step", 1LL));
    ORT_ENFORCE(step_ > 0, "step must greater than zero!");

    int64_t temp_size;
    ORT_ENFORCE(info.GetAttr("size", &temp_size).IsOK());
    size_ = SafeInt<int>(temp_size);
  }

  ~UnfoldTensor() = default;

  Status Compute(OpKernelContext* context) const override;

 private:
  int dim_;
  int size_;
  int step_;
};

}  // namespace contrib
}  // namespace onnxruntime
