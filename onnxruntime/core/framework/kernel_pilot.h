// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <functional>
#include <optional>

#include "core/framework/kernel_pilot_moe_expert_cache.h"
#include "core/framework/kernel_pilot_moe_expert_selection.h"
#include "core/framework/kernel_pilot_moe_expert_usage.h"

namespace onnxruntime {

class KernelPilotMoeExpertState;
class KernelPilot;
class OpKernel;

// Registers an in-tree kernel callback that runs after session-global piloting state is finalized.
// Keeping this callback outside OpKernel preserves the shared-provider OpKernel vtable.
class KernelPilotInitializer {
 public:
  using Callback = std::function<Status(KernelPilot*)>;

  KernelPilotInitializer() = default;
  ~KernelPilotInitializer();

  void Register(const OpKernel* kernel, Callback callback);
  static Status Run(const OpKernel* kernel, KernelPilot* pilot);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelPilotInitializer);

 private:
  const OpKernel* kernel_{nullptr};
};

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
  double MoeExpertCounterProbabilityWeight() const noexcept { return moe_expert_counter_probability_weight_; }

  void RecordMoeDemandSwaps(size_t missing, size_t swapped, bool all_required_on_cuda) {
    moe_demand_swaps_ = MoeDemandSwaps{missing, swapped, all_required_on_cuda};
  }

 private:
  friend class KernelPilotMoeExpertState;
  void FinishRegistration() noexcept;
  Status CompleteMoeUsageInternal();
  void SetMoeCudaExperts(gsl::span<int> expert_ids) noexcept {
    moe_cuda_experts_ = expert_ids;
  }
  IKernelPilotMoeExpertCache* GetMoeExpertCache() const noexcept { return moe_cache_; }

  KernelPilotMoeExpertState& moe_expert_state_;
  const OpKernel* kernel_;
  KernelPilotMoeExpertSelection moe_;
  gsl::span<int> moe_cuda_experts_;
  IKernelPilotMoeExpertCache* moe_cache_{nullptr};
  IKernelPilotMoeExpertUsage* moe_usage_{nullptr};
  gsl::span<const double> moe_counters_;
  Status (*complete_moe_usage_)(KernelPilot&);
  bool deferred_invocation_succeeded_{false};
  size_t moe_expert_count_{0};
  double moe_expert_counter_alpha_{0.75};
  double moe_expert_counter_probability_weight_{0.0};
  struct MoeDemandSwaps {
    size_t missing, swapped;
    bool all_required_on_cuda;
  };
  std::optional<MoeDemandSwaps> moe_demand_swaps_;
};

}  // namespace onnxruntime
