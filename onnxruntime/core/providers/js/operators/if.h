// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>

#include "core/providers/js/js_kernel.h"
#include "core/common/common.h"
#include "core/providers/cpu/controlflow/if.h"

namespace onnxruntime {
class SessionState;

namespace js {

// Use the CPU implementation for the logic
class If final : public onnxruntime::If {
 public:
  If(const OpKernelInfo& info) : onnxruntime::If(info) {}

  Status Compute(OpKernelContext* ctx) const override;
};
}  // namespace js
}  // namespace onnxruntime
