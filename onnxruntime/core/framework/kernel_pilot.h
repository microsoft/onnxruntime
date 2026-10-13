// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_pilot_moe_expert_selection.h"
#include "core/framework/kernel_pilot_moe_expert_usage.h"

namespace onnxruntime {

class KernelPilotMoeExpertState;
class OpKernel;

// Per-kernel piloting state: information a kernel implementation needs from, or reports back
// to, the session between invocations, beyond its normal tensor inputs/outputs. A session owns
// one KernelPilot per registered kernel that requires it; OpKernelContext exposes the current
// kernel's pilot through GetKernelPilot().
class KernelPilot {
 public:
  KernelPilot(KernelPilotMoeExpertState& moe_expert_state, const OpKernel* kernel);

  IKernelPilotMoeExpertSelection& Moe() noexcept { return moe_; }
  const IKernelPilotMoeExpertSelection& Moe() const noexcept { return moe_; }

  // Returns the local expert IDs selected for static CUDA residency. An empty span is valid.
  Status GetMoeCudaExperts(gsl::span<const int>& expert_ids) const {
    expert_ids = moe_cuda_experts_;
    return Status::OK();
  }

  // Commits data collected by this pilot after a successful kernel invocation. A pilot that
  // was only queried, without starting an invocation, has nothing to commit.
  Status RecordUsage();

  Status AttachMoeExpertUsage(IKernelPilotMoeExpertUsage* usage) {
    ORT_RETURN_IF_NOT(usage, "MoE expert usage must not be null.");
    ORT_RETURN_IF(moe_usage_ != nullptr && moe_usage_ != usage, "A different MoE usage collector is already attached.");
    moe_usage_ = usage;
    return Status::OK();
  }

  // Indirection keeps this call usable across the shared-provider boundary.
  Status CompleteMoeUsage() { return complete_moe_usage_(*this); }
  gsl::span<const double> MoeExpertCounters() const noexcept { return moe_counters_; }
  double MoeExpertCounterAlpha() const noexcept { return moe_expert_counter_alpha_; }
  double MoeExpertCounterProbabilityWeight() const noexcept { return 0.0; }

 private:
  friend class KernelPilotMoeExpertState;
  void FinishRegistration() noexcept;
  Status CompleteMoeUsageInternal();
  void SetMoeCudaExperts(gsl::span<const int> expert_ids) noexcept {
    moe_cuda_experts_ = expert_ids;
  }

  KernelPilotMoeExpertState& moe_expert_state_;
  const OpKernel* kernel_;
  KernelPilotMoeExpertSelection moe_;
  gsl::span<const int> moe_cuda_experts_;
  IKernelPilotMoeExpertUsage* moe_usage_{nullptr};
  gsl::span<const double> moe_counters_;
  Status (*complete_moe_usage_)(KernelPilot&);
  bool deferred_invocation_succeeded_{false};
  double moe_expert_counter_alpha_{0.9};
};

}  // namespace onnxruntime
