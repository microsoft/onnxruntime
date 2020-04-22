// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

void QLinearLookupTableTransform(const uint8_t* x, const uint8_t* table, uint8_t* y, size_t n);

template <typename T>
class QLinearLeakyRelu final : public OpKernel {
 public:
  QLinearLeakyRelu(const OpKernelInfo& info)
      : OpKernel(info), alpha_(info.GetAttrOrDefault("alpha", 0.01f)) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  const float alpha_;
};

}  // namespace contrib
}  // namespace onnxruntime
