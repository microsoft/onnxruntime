// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <istream>
#include <map>
#include <string>
#include <string_view>
#include <utility>

#include <gsl/gsl>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {

class OpKernel;

// Host-side state only. CUDA allocations and placement policy do not belong here.
class MoeExpertState {
 public:
  MoeExpertState() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MoeExpertState);

  struct NodeCounters {
    std::string node_type;
    InlinedVector<double> counters;
  };
  using Key = std::pair<std::string, size_t>;
  using Snapshot = std::map<Key, NodeCounters>;

  Status SetCounterParameters(double alpha, double beta);
  Status RegisterNode(const OpKernel* kernel, std::string_view graph_scope, size_t node_index,
                      std::string_view node_type, size_t expert_count);
  Status Load(std::istream& input);
  Status FinalizeInitialization();
  Status BeginRun() const;
  void EndRun() const;
  Status RecordUsage(const OpKernel* kernel, gsl::span<const int> used_expert_ids);
  // During a Run, only the executing kernel may read its own counters.
  Status GetCounters(const OpKernel* kernel, InlinedVector<double>& counters) const;
  Status GetExpertId(const OpKernel* kernel, int expert_id, size_t& global_expert_id) const;
  // Call only while no Run is active.
  Snapshot GetSnapshot() const;
  size_t TotalExpertCount() const noexcept { return counters_.size(); }

 private:
  struct ExpertRange {
    size_t begin;
    size_t count;
  };
  struct NodeInfo {
    std::string node_type;
    ExpertRange experts;
  };
  struct Counter {
    double value;
    double next_value;
  };

  std::map<Key, NodeInfo> nodes_;
  InlinedHashMap<const OpKernel*, ExpertRange> kernel_experts_;
  InlinedHashMap<std::pair<const OpKernel*, int>, size_t> expert_ids_;
  InlinedVector<Counter> counters_;
  double alpha_{1.0};
  double beta_{1.0};
  bool initialized_{false};
  // Reject overlapping runs once per Run, not once per kernel or expert.
  mutable std::atomic_flag run_active_ = ATOMIC_FLAG_INIT;
};

}  // namespace onnxruntime
