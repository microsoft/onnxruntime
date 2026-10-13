// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_pilot.h"

#include <cmath>

#include "core/framework/kernel_pilot_moe_expert_state.h"

namespace onnxruntime {

KernelPilot::KernelPilot(KernelPilotMoeExpertState& moe_expert_state, const OpKernel* kernel)
    : moe_expert_state_(moe_expert_state),
      kernel_(kernel),
      complete_moe_usage_([](KernelPilot& pilot) { return pilot.CompleteMoeUsageInternal(); }),
      moe_expert_counter_alpha_(moe_expert_state.alpha_),
      moe_expert_counter_probability_weight_(moe_expert_state.CounterProbabilityWeight()) {
}

Status KernelPilot::RecordUsage() {
  if (moe_usage_ != nullptr && moe_usage_->HasPendingUsage()) {
    if (!deferred_invocation_succeeded_) {
      std::lock_guard<std::mutex> lock(moe_expert_state_.counter_logging_mutex_);
      moe_expert_state_.deferred_usage_.push_back(this);
    }
    deferred_invocation_succeeded_ = true;
    return Status::OK();
  }
  if (!moe_.HasPendingInvocation()) {
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(moe_expert_state_.RecordUsage(kernel_));
  moe_.FinishInvocation();
  return Status::OK();
}

Status KernelPilot::CompleteMoeUsageInternal() {
  if (moe_usage_ == nullptr || !moe_usage_->HasPendingUsage()) {
    return Status::OK();
  }
  KernelPilotMoeExpertUsageSnapshot snapshot;
  const Status completion = moe_usage_->CompleteUsage(snapshot);
  const bool commit = deferred_invocation_succeeded_;
  deferred_invocation_succeeded_ = false;
  auto finish = gsl::finally([&]() { moe_.FinishInvocation(); });
  ORT_RETURN_IF_ERROR(completion);
  if (!commit) {
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(snapshot.selected_probabilities.size() == moe_.ExpertCount(),
                    "MoE GPU selection size does not match registration.");
  ORT_RETURN_IF_NOT(snapshot.counters.size() == moe_.ExpertCount(),
                    "MoE GPU counter size does not match registration.");
  ORT_RETURN_IF_NOT(snapshot.first_selected_routes.size() == moe_.ExpertCount(),
                    "MoE GPU selection order size does not match registration.");
  InlinedVector<int> ids;
  InlinedVector<float> probabilities;
  for (size_t expert = 0; expert < snapshot.selected_probabilities.size(); ++expert) {
    const float probability = snapshot.selected_probabilities[expert];
    ORT_RETURN_IF_NOT(std::isfinite(probability) && (probability == -1.0f || (probability >= 0.0f && probability <= 1.0f)),
                      "Invalid MoE GPU routing probability.");
    if (probability >= 0.0f) {
      ORT_RETURN_IF(snapshot.first_selected_routes[expert] < 0, "Missing MoE GPU selection order.");
      ids.push_back(static_cast<int>(expert));
    }
  }
  std::sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
    return snapshot.first_selected_routes[static_cast<size_t>(lhs)] <
           snapshot.first_selected_routes[static_cast<size_t>(rhs)];
  });
  for (int expert : ids) {
    probabilities.push_back(snapshot.selected_probabilities[static_cast<size_t>(expert)]);
  }
  if (moe_expert_counter_probability_weight_ > 0.0) {
    ORT_RETURN_IF_ERROR(moe_.CollectWeighted(ids, probabilities));
  } else {
    ORT_RETURN_IF_ERROR(moe_.Collect(ids));
  }
  return moe_expert_state_.RecordUsage(kernel_, snapshot.counters);
}

void KernelPilot::FinishRegistration() noexcept {
  moe_expert_count_ = moe_.ExpertCount();
  moe_.FinishInvocation();
}

}  // namespace onnxruntime
