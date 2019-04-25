// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/autopad_type.h"

namespace onnxruntime {

class MaxUnpool : public OpKernel {
 public:
  MaxUnpool(const OpKernelInfo& info) : OpKernel(info) {
    num_inputs_ = OpKernel::Node().InputDefs().size();
    bool has_output_shape = num_inputs_ == 3;
    bool has_kernal_shape = info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK() &&
      !kernel_shape_.empty();
    ORT_ENFORCE(has_output_shape || has_kernal_shape,
      "Need to compute output shape but kernel shape is not set.");

    bool has_pads = info.GetAttrs<int64_t>("pads", pads_).IsOK() && !pads_.empty();
    if (has_kernal_shape && !has_pads) {
      // setup defaults.
      // we need pads_ to either validate or compute the output shape
      pads_.resize(kernel_shape_.size() * 2, 0);
    }

    bool has_strides = info.GetAttrs<int64_t>("strides", strides_).IsOK() && !strides_.empty();
    if (has_kernal_shape && !has_strides) {
      // we need strides_ to either validate or compute the output shape
      strides_.resize(kernel_shape_.size(), 1);
      }

    for (size_t dim = 0; dim < kernel_shape_.size(); ++dim) {
      ORT_ENFORCE(kernel_shape_[dim] > 0);
      ORT_ENFORCE(pads_[dim] < kernel_shape_[dim] && pads_[dim + kernel_shape_.size()] < kernel_shape_[dim],
                  "Pad should be smaller than kernel.");
    }

    ORT_ENFORCE(strides_.size() == kernel_shape_.size());

    if (has_kernal_shape)
    {
      // Add 4 pad values (0) for batch and channel dimensions
      pads_.insert(pads_.begin(), {0, 0});
      pads_.insert(pads_.begin() + 2 + kernel_shape_.size(), {0, 0});

      // Separate out any negative pads_ into the slices_ array
      slices_.resize(pads_.size(), 0);
      for (size_t index = 0; index < pads_.size(); index++) {
        if (pads_[index] < 0) {
          slices_[index] = pads_[index];
          pads_[index] = 0;
        }
      }
    }
  }

  ~MaxUnpool() override{};

  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> slices_;  // All of the negative padding values are separated out into slices_
  int64_t num_inputs_;
};

}  // namespace onnxruntime
