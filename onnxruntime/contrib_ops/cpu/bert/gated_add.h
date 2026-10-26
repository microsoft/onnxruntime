// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// output = X + round_to_T(Y * gate), with gate broadcast across the last dimension.
template <typename T>
class GatedAdd final : public OpKernel {
 public:
  explicit GatedAdd(const OpKernelInfo& info) : OpKernel(info) {
    const int64_t fp32 = info.GetAttrOrDefault<int64_t>("sigmoid_in_fp32", 1);
    ORT_ENFORCE(fp32 == 0 || fp32 == 1, "GatedAdd sigmoid_in_fp32 must be zero or one");
  }
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
