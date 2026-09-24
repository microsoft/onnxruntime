// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_pilot_moe_expert_selection.h"

namespace onnxruntime {

// Per-kernel piloting state: information a kernel implementation needs from, or reports back
// to, the session between invocations, beyond its normal tensor inputs/outputs. A session owns
// one KernelPilot per registered kernel that requires it; OpKernelContext exposes the current
// kernel's pilot through GetKernelPilot(). The class itself carries no kernel-specific logic;
// each kernel family gets a separate type holding the information it actually needs.
class KernelPilot {
 public:
  IKernelPilotMoeExpertSelection& Moe() noexcept { return moe_; }
  const IKernelPilotMoeExpertSelection& Moe() const noexcept { return moe_; }

 private:
  KernelPilotMoeExpertSelection moe_;
};

}  // namespace onnxruntime
