// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class ConstantFill final : public OpKernel {
 public:
  ConstantFill(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr("dtype", &dtype_).IsOK()) {
      ONNXRUNTIME_THROW("Missing 'dtype' attribute value");
    }
    if (!info.GetAttr("input_as_shape", &input_as_shape_).IsOK()) {
      ONNXRUNTIME_THROW("Missing 'input_as_shape' attribute value");
    }
    if (input_as_shape_) {
      if (!info.GetAttrs("extra_shape", extra_shape_).IsOK()) {
        ONNXRUNTIME_THROW("Missing 'extra_shape' attribute value");
      }
    } else {
      if (!info.GetAttrs("shape", shape_).IsOK()) {
        ONNXRUNTIME_THROW("Missing 'shape' attribute value");
      }
    }
    if (!info.GetAttr("value", &value_).IsOK()) {
      ONNXRUNTIME_THROW("Missing 'value' attribute value");
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  std::vector<int64_t> DimsFromInput(const Tensor* t1) const;

  template <typename T>
  Status ComputeImpl(OpKernelContext* context) const;

  int64_t dtype_;
  std::vector<int64_t> extra_shape_;
  int64_t input_as_shape_;
  std::vector<int64_t> shape_;
  float value_;
};

}  // namespace onnxruntime
