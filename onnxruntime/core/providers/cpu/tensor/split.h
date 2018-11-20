// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <numeric>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class Split final : public OpKernel {
 public:
  Split(const OpKernelInfo& info) : OpKernel(info) {
    // required with default of 0
    if (!info.GetAttr("axis", &axis_).IsOK())
      ONNXRUNTIME_THROW("Missing 'axis' attribute value");

    // optional
    if (info.GetAttrs("split", split_sizes_).IsOK()) {
      split_size_sum_ = std::accumulate(split_sizes_.cbegin(), split_sizes_.cend(), 0LL);
      ONNXRUNTIME_ENFORCE(std::all_of(split_sizes_.cbegin(), split_sizes_.cend(), [](int64_t value) { return value > 0; }),
                  "Invalid value in 'split' attribute. All values must be > 0");
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext& context, const Tensor& input) const;

  int64_t axis_;
  std::vector<int64_t> split_sizes_;
  int64_t split_size_sum_ = 0;
};

}  // namespace onnxruntime
