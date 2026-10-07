// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>

#include "core/framework/kernel_pilot_moe_expert_cache.h"
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

  Status AttachMoeExpertCache(IKernelPilotMoeExpertCache* cache) {
    ORT_RETURN_IF_NOT(cache, "MoE expert cache must not be null.");
    ORT_RETURN_IF(moe_cache_ != nullptr && moe_cache_ != cache,
                  "A different MoE expert cache is already attached.");
    moe_cache_ = cache;
    return Status::OK();
  }

  Status PublishMoeExpertSwap(int cuda_expert_id, int cpu_expert_id) {
    const auto cuda_expert =
        std::find(moe_cuda_experts_.begin(), moe_cuda_experts_.end(), cuda_expert_id);
    ORT_RETURN_IF(cuda_expert == moe_cuda_experts_.end(),
                  "MoE swap tried to evict a non-resident CUDA expert: ", cuda_expert_id);
    ORT_RETURN_IF(cpu_expert_id < 0 ||
                      static_cast<size_t>(cpu_expert_id) >= moe_expert_count_ ||
                      std::find(moe_cuda_experts_.begin(), moe_cuda_experts_.end(), cpu_expert_id) !=
                          moe_cuda_experts_.end(),
                  "MoE swap tried to publish an invalid CPU expert: ", cpu_expert_id);
    *cuda_expert = cpu_expert_id;
    std::sort(moe_cuda_experts_.begin(), moe_cuda_experts_.end());
    return Status::OK();
  }

  // Commits data collected by this pilot after a successful kernel invocation. A pilot that
  // was only queried, without starting an invocation, has nothing to commit.
  Status RecordUsage();

 private:
  friend class KernelPilotMoeExpertState;
  void FinishRegistration() noexcept;
  void SetMoeCudaExperts(gsl::span<int> expert_ids) noexcept {
    moe_cuda_experts_ = expert_ids;
  }
  IKernelPilotMoeExpertCache* GetMoeExpertCache() const noexcept { return moe_cache_; }

  KernelPilotMoeExpertState& moe_expert_state_;
  const OpKernel* kernel_;
  KernelPilotMoeExpertSelection moe_;
  gsl::span<int> moe_cuda_experts_;
  IKernelPilotMoeExpertCache* moe_cache_{nullptr};
  size_t moe_expert_count_{0};
};

}  // namespace onnxruntime
