// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime::contrib {

template <typename T>
class HyperConnectionPreMix final : public OpKernel {
 public:
  explicit HyperConnectionPreMix(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t num_branches_;
  float reduction_scale_;
};

}  // namespace onnxruntime::contrib
