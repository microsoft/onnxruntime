// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/concat.h"

namespace onnxruntime {

class ConcatFromSequence final : public OpKernel, public ConcatBase {
 public:
  explicit ConcatFromSequence(const OpKernelInfo& info) : OpKernel(info), ConcatBase(info, true) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
