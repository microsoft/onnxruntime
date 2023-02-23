// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class FakeQuant final : public OpKernel {
 public:
  FakeQuant(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault("quant_min", &quant_min_, static_cast<decltype(quant_min_)>(0));
    info.GetAttrOrDefault("quant_max", &quant_max_, static_cast<decltype(quant_max_)>(255));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t quant_min_;
  int64_t quant_max_;
};

template <typename T>
class FakeQuantGrad final : public OpKernel {
 public:
  FakeQuantGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
