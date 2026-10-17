// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_pilot_moe_expert_selection.h"

#include <algorithm>
#include <cmath>

namespace onnxruntime {

KernelPilotMoeExpertSelection::KernelPilotMoeExpertSelection() = default;

Status KernelPilotMoeExpertSelection::BeginInvocation(size_t expert_count) {
  ORT_RETURN_IF(expert_count == 0, "MoE expert selection requires a positive expert count.");
  expert_count_ = expert_count;
  // Unlike clear(), erasing preserves InlinedVector's allocated storage.
  selected_experts_.erase(selected_experts_.begin(), selected_experts_.end());
  selected_expert_probabilities_.erase(selected_expert_probabilities_.begin(),
                                       selected_expert_probabilities_.end());
  selected_experts_.reserve(expert_count);
  selected_expert_probabilities_.reserve(expert_count);
  has_routing_probabilities_ = false;
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
  ORT_RETURN_IF(has_routing_probabilities_,
                "Cannot mix weighted and unweighted MoE expert collection in one invocation.");
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

Status KernelPilotMoeExpertSelection::CollectWeighted(
    gsl::span<const int> expert_ids,
    gsl::span<const float> routing_probabilities) {
  ORT_RETURN_IF_NOT(IsInitialized(), "MoE expert selection was not initialized.");
  ORT_RETURN_IF(!selected_experts_.empty() && !has_routing_probabilities_,
                "Cannot mix weighted and unweighted MoE expert collection in one invocation.");
  ORT_RETURN_IF_NOT(expert_ids.size() == routing_probabilities.size(),
                    "MoE expert IDs and routing probabilities must have the same size.");
  for (size_t i = 0; i < expert_ids.size(); ++i) {
    ORT_RETURN_IF(expert_ids[i] < 0 || static_cast<size_t>(expert_ids[i]) >= expert_count_,
                  "MoE expert index out of range: ", expert_ids[i]);
    ORT_RETURN_IF_NOT(std::isfinite(routing_probabilities[i]) &&
                          routing_probabilities[i] >= 0.0f && routing_probabilities[i] <= 1.0f,
                      "MoE expert routing probabilities must be finite and in [0, 1].");
  }
  for (size_t i = 0; i < expert_ids.size(); ++i) {
    const int expert = expert_ids[i];
    const auto existing = std::find(selected_experts_.begin(), selected_experts_.end(), expert);
    if (existing == selected_experts_.end()) {
      selected_experts_.push_back(expert);
      selected_expert_probabilities_.push_back(routing_probabilities[i]);
    } else {
      const size_t index = static_cast<size_t>(std::distance(selected_experts_.begin(), existing));
      selected_expert_probabilities_[index] =
          std::max(selected_expert_probabilities_[index], routing_probabilities[i]);
    }
  }
  has_routing_probabilities_ = true;
  return Status::OK();
}

Status KernelPilotMoeExpertSelection::GetSelectedExperts(gsl::span<const int>& expert_ids) const {
  ORT_RETURN_IF_NOT(IsInitialized(), "MoE expert selection was not initialized.");
  expert_ids = selected_experts_;
  return Status::OK();
}

Status KernelPilotMoeExpertSelection::GetSelectedExpertProbabilities(
    gsl::span<const float>& routing_probabilities) const {
  ORT_RETURN_IF_NOT(IsInitialized(), "MoE expert selection was not initialized.");
  routing_probabilities = has_routing_probabilities_ ? gsl::make_span(selected_expert_probabilities_)
                                                     : gsl::span<const float>{};
  return Status::OK();
}

bool KernelPilotMoeExpertSelection::HasPendingInvocation() const noexcept {
  return invocation_pending_;
}

void KernelPilotMoeExpertSelection::FinishInvocation() noexcept {
  invocation_pending_ = false;
}

}  // namespace onnxruntime
