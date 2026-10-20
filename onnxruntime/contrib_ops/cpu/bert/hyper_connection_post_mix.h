// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime::contrib {

template <typename T>
class HyperConnectionPostMix final : public OpKernel {
 public:
  explicit HyperConnectionPostMix(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t num_branches_;
};

}  // namespace onnxruntime::contrib
