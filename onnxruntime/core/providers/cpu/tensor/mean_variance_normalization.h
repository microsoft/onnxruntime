// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class MeanVarianceNormalization : public OpKernel {
 public:
  MeanVarianceNormalization(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  const bool normalize_variance_;
  const InlinedVector<int64_t> axes_;
};
}  // namespace onnxruntime
