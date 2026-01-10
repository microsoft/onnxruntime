// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {

class MaxUnpool : public OpKernel {
 public:
  MaxUnpool(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttrs<int64_t>("kernel_shape", kernel_shape_));

    num_inputs_ = OpKernel::Node().InputDefs().size();

    if (num_inputs_ == 3 && !pads_.empty()) {
      // ignore pads attribute value
    }

    // setup defaults.
    if (!info.GetAttrs<int64_t>("pads", pads_).IsOK() || pads_.empty()) {
      pads_.resize(kernel_shape_.size() * 2, 0);
    }

    if (!info.GetAttrs<int64_t>("strides", strides_).IsOK() || strides_.empty()) {
      strides_.resize(kernel_shape_.size(), 1);
    }

    for (size_t dim = 0; dim < kernel_shape_.size(); ++dim) {
      ORT_ENFORCE(kernel_shape_[dim] > 0);
      ORT_ENFORCE(pads_[dim] < kernel_shape_[dim] && pads_[dim + kernel_shape_.size()] < kernel_shape_[dim],
                  "Pad should be smaller than kernel.");
    }

    ORT_ENFORCE(strides_.size() == kernel_shape_.size());
  }

  ~MaxUnpool() override = default;

  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  int64_t num_inputs_;
};

}  // namespace onnxruntime
