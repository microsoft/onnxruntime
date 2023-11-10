// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T>
class DequantizeLinear final : public OpKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

template <typename T>
class QuantizeLinear final : public OpKernel {
 public:
  QuantizeLinear(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
    if (!info.GetAttr<int64_t>("saturate", &saturate_).IsOK()) {
      saturate_ = 1;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  int64_t saturate_;
};
}  // namespace onnxruntime
