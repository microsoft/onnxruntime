// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class QLinearGlobalAveragePool final : public OpKernel {
 public:
  QLinearGlobalAveragePool(const OpKernelInfo& info) : OpKernel(info) {
    channels_last_ = (info.GetAttrOrDefault<int64_t>("channels_last", static_cast<int64_t>(0)) != 0);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool channels_last_;
};

Status ComputeQLinearGlobalAvgPool(
    const uint8_t* x,
    float x_scale,
    uint8_t x_zero_point,
    uint8_t* y,
    float y_scale,
    uint8_t y_zero_point,
    int64_t N,
    int64_t C,
    int64_t image_size,
    bool channels_last,
    concurrency::ThreadPool* tp);

}  // namespace contrib
}  // namespace onnxruntime
