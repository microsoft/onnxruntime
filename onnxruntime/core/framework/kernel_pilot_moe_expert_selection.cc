// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_pilot_moe_expert_selection.h"

#include <algorithm>

namespace onnxruntime {

KernelPilotMoeExpertSelection::KernelPilotMoeExpertSelection() = default;

Status KernelPilotMoeExpertSelection::BeginInvocation(size_t expert_count) {
  ORT_RETURN_IF(expert_count == 0, "MoE expert selection requires a positive expert count.");
  expert_count_ = expert_count;
  // Unlike clear(), erasing preserves InlinedVector's allocated storage.
  selected_experts_.erase(selected_experts_.begin(), selected_experts_.end());
  selected_experts_.reserve(expert_count);
  invocation_pending_ = true;
  return Status::OK();
}

bool KernelPilotMoeExpertSelection::IsInitialized() const noexcept {
  return expert_count_ != 0;
}

size_t KernelPilotMoeExpertSelection::ExpertCount() const noexcept {
  return expert_count_;
}

Status KernelPilotMoeExpertSelection::Collect(gsl::span<const int> expert_ids) {
  ORT_RETURN_IF_NOT(IsInitialized(), "MoE expert selection was not initialized.");
  for (int expert : expert_ids) {
    ORT_RETURN_IF(expert < 0 || static_cast<size_t>(expert) >= expert_count_,
                  "MoE expert index out of range: ", expert);
  }
  for (int expert : expert_ids) {
    if (std::find(selected_experts_.begin(), selected_experts_.end(), expert) == selected_experts_.end()) {
      selected_experts_.push_back(expert);
    }
  }
  return Status::OK();
}

Status KernelPilotMoeExpertSelection::GetSelectedExperts(gsl::span<const int>& expert_ids) const {
  ORT_RETURN_IF_NOT(IsInitialized(), "MoE expert selection was not initialized.");
  expert_ids = selected_experts_;
  return Status::OK();
}

bool KernelPilotMoeExpertSelection::HasPendingInvocation() const noexcept {
  return invocation_pending_;
}

void KernelPilotMoeExpertSelection::FinishInvocation() noexcept {
  invocation_pending_ = false;
}

}  // namespace onnxruntime
