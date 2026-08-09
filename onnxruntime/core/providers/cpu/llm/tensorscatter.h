// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class TensorScatter final : public OpKernel {
 public:
  TensorScatter(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  bool circular_;
};

}  // namespace onnxruntime
