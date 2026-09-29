// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename T>
class Col2Im final : public OpKernel {
 public:
  explicit Col2Im(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttrs("strides", strides_).IsOK())
      ORT_ENFORCE(strides_.empty());
    if (!info.GetAttrs("dilations", dilations_).IsOK())
      ORT_ENFORCE(dilations_.empty());
    if (!info.GetAttrs("pads", pads_).IsOK())
      ORT_ENFORCE(pads_.empty());
    for (auto s : strides_) {
      ORT_ENFORCE(s > 0, "All stride values must be positive, got: ", s);
    }
    for (auto d : dilations_) {
      ORT_ENFORCE(d > 0, "All dilation values must be positive, got: ", d);
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  TensorShapeVector pads_;
  TensorShapeVector dilations_;
  TensorShapeVector strides_;
};

}  // namespace onnxruntime
