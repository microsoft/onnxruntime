// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cpu/controlflow/loop.h"

namespace onnxruntime {
namespace cuda {

// Use the CPU implementation for the logic
class Loop final : public onnxruntime::Loop {
 public:
  Loop(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;
};
}  // namespace cuda
}  // namespace onnxruntime
