// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_pilot.h"

#include "core/framework/kernel_pilot_moe_expert_state.h"

namespace onnxruntime {

KernelPilot::KernelPilot(KernelPilotMoeExpertState& moe_expert_state, const OpKernel* kernel)
    : moe_expert_state_(moe_expert_state), kernel_(kernel) {
}

Status KernelPilot::RecordUsage() {
  if (!moe_.HasPendingInvocation()) {
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(moe_expert_state_.RecordUsage(kernel_));
  moe_.FinishInvocation();
  return Status::OK();
}

void KernelPilot::FinishRegistration() noexcept {
  moe_.FinishInvocation();
}

}  // namespace onnxruntime
