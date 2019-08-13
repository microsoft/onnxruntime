// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/conv_base.h"
#include "core/util/gemmlowp_common.h"

namespace onnxruntime {
class QLinearConv : public OpKernel, public ConvBase {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), ConvBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;  
};
}  // namespace onnxruntime
