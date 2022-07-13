// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <numeric>

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {

class SplitBase {
 public:
  /*
   * \param num_outputs must >=0
   */
  Status PrepareForCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                           int& after_dims_including_split_axis, int& after_dims_excluding_split,
                           std::vector<int64_t>& split_sizes) const;

 protected:
  SplitBase(const OpKernelInfo& info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);

    size_t numInputs = info.GetInputCount();
    if (numInputs == 1) {
      // optional
      if (info.GetAttrs("split", split_sizes_).IsOK()) {
        split_size_sum_ = std::accumulate(split_sizes_.cbegin(), split_sizes_.cend(), 0LL);
        ORT_ENFORCE(std::all_of(split_sizes_.cbegin(), split_sizes_.cend(), [](int64_t value) { return value >= 0; }),
                    "Invalid value in 'split' attribute. All values must be > 0");
      }
    }
  }

  int64_t axis_;
  std::vector<int64_t> split_sizes_;
  int64_t split_size_sum_ = -1;
};

class Split final : public OpKernel, public SplitBase {
 public:
  Split(const OpKernelInfo& info) : OpKernel(info), SplitBase(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext& context, const Tensor& input) const;
};

}  // namespace onnxruntime
