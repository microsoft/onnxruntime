// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cpu/controlflow/loop.h"

namespace onnxruntime {
namespace cuda {

// Use the CPU implementation for the logic
class Loop final : public OpKernel {
 public:
  Loop(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

 private:
  std::unique_ptr<OpKernel> cpu_loop_;
};
}  // namespace cuda
}  // namespace onnxruntime
