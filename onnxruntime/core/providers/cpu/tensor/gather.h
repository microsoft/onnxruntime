// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "gatherbase.h"

namespace onnxruntime {

class Gather final : public OpKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : OpKernel(info), GatherBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace onnxruntime
