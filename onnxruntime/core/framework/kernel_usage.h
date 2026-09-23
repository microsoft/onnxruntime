// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {

// Collects the union of local expert IDs across one kernel invocation, including tiled routing.
class KernelUsage {
 public:
  KernelUsage() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelUsage);

  Status BeginInvocation(size_t expert_count) {
    ORT_RETURN_IF(expert_count == 0, "Kernel usage collection requires a positive expert count.");
    used_.assign(expert_count, 0);
    // Unlike clear(), erasing preserves InlinedVector's allocated storage.
    selected_experts_.erase(selected_experts_.begin(), selected_experts_.end());
    selected_experts_.reserve(expert_count);
    return Status::OK();
  }

  bool IsInitialized() const noexcept { return !used_.empty(); }
  size_t ExpertCount() const noexcept { return used_.size(); }

  Status Collect(gsl::span<const int> expert_ids) {
    ORT_RETURN_IF_NOT(IsInitialized(), "Kernel usage collection was not initialized.");
    for (int expert : expert_ids) {
      ORT_RETURN_IF(expert < 0 || static_cast<size_t>(expert) >= used_.size(),
                    "Kernel usage expert index out of range: ", expert);
    }
    for (int expert : expert_ids) {
      if (!used_[expert]) {
        used_[expert] = 1;
        selected_experts_.push_back(expert);
      }
    }
    return Status::OK();
  }

  Status GetSelectedExperts(gsl::span<const int>& expert_ids) const {
    ORT_RETURN_IF_NOT(IsInitialized(), "Kernel usage collection was not initialized.");
    expert_ids = selected_experts_;
    return Status::OK();
  }

 private:
  InlinedVector<uint8_t> used_;
  InlinedVector<int> selected_experts_;
};

}  // namespace onnxruntime
