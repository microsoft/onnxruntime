// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {

// Per-kernel piloting state: information a kernel implementation needs from, or reports back
// to, the session between invocations, beyond its normal tensor inputs/outputs. A session owns
// one KernelPilot per registered kernel that requires it; OpKernelContext exposes the current
// kernel's pilot through GetKernelPilot(). The class itself carries no kernel-specific logic;
// each kernel family gets its own nested type holding the information it actually needs.
class KernelPilot {
 public:
  // MoE-specific piloting information: collects the union of local expert IDs selected across
  // one kernel invocation, including tiled routing.
  class MoeExpertSelection {
   public:
    MoeExpertSelection() = default;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MoeExpertSelection);

    Status BeginInvocation(size_t expert_count) {
      ORT_RETURN_IF(expert_count == 0, "MoE expert selection requires a positive expert count.");
      expert_count_ = expert_count;
      // Unlike clear(), erasing preserves InlinedVector's allocated storage.
      selected_experts_.erase(selected_experts_.begin(), selected_experts_.end());
      selected_experts_.reserve(expert_count);
      return Status::OK();
    }

    bool IsInitialized() const noexcept { return expert_count_ != 0; }
    size_t ExpertCount() const noexcept { return expert_count_; }

    // expert_count is small (the number of experts configured for this MoE node), so a linear
    // scan of the already-selected list is cheaper than maintaining a separate expert_count-sized
    // membership mask.
    Status Collect(gsl::span<const int> expert_ids) {
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

    Status GetSelectedExperts(gsl::span<const int>& expert_ids) const {
      ORT_RETURN_IF_NOT(IsInitialized(), "MoE expert selection was not initialized.");
      expert_ids = selected_experts_;
      return Status::OK();
    }

   private:
    size_t expert_count_{0};
    InlinedVector<int> selected_experts_;
  };

  MoeExpertSelection& Moe() noexcept { return moe_; }
  const MoeExpertSelection& Moe() const noexcept { return moe_; }

 private:
  MoeExpertSelection moe_;
};

}  // namespace onnxruntime
