// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_OPS
// Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
// 2). this is needed by inference for other purpose.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/gather.h"

namespace onnxruntime {
namespace contrib {

class ShrunkenGatherCommon {
 public:
  void CheckInput(const Tensor* input_tensor, const Tensor* indices_tensor, int64_t axis_in) const;
};

class ShrunkenGather final : public onnxruntime::Gather, public ShrunkenGatherCommon {
 public:
  ShrunkenGather(const OpKernelInfo& info) : Gather(info) {}

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace contrib
}  // namespace onnxruntime

#endif
