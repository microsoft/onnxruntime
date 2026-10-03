// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_pilot_moe_expert_selection.h"

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

 private:
  friend class KernelPilotMoeExpertState;
  void FinishRegistration() noexcept;
  void SetMoeCudaExperts(gsl::span<const int> expert_ids) noexcept {
    moe_cuda_experts_ = expert_ids;
  }

  KernelPilotMoeExpertState& moe_expert_state_;
  const OpKernel* kernel_;
  KernelPilotMoeExpertSelection moe_;
  gsl::span<const int> moe_cuda_experts_;
};

}  // namespace onnxruntime
