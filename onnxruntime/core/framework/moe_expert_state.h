// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <istream>
#include <string>
#include <string_view>
#include <utility>

#include <gsl/gsl>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/kernel_pilot.h"

namespace onnxruntime {

class OpKernel;

// Host-side state only. CUDA allocations and placement policy do not belong here.
class MoeExpertState {
 public:
  MoeExpertState() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MoeExpertState);

  Status SetCounterParameters(double alpha, double beta);
  Status RegisterNode(const OpKernel* kernel, std::string_view graph_scope, size_t node_index,
                      std::string_view node_type, size_t expert_count);

  // Loads initial counters after node registration and before FinalizeInitialization().
  // UTF-8 text format: the first line must be exactly "moe_expert_state 1", followed by
  // one whitespace-separated record per line:
  //   "graph_scope" node_index node_type expert_id counter_value
  // Example:
  //   moe_expert_state 1
  //   "main" 0 MoE 2 12
  //   "main/4/11:then_branch" 0 QMoE 1 3.5
  //
  // graph_scope uses std::quoted escaping for quotes/backslashes; unquoted scopes without
  // whitespace are also accepted. The root scope is "main"; subgraphs append
  // /<parent-node-index>/<attribute-name-length>:<attribute-name> (length in bytes).
  // node_index identifies the registered node in the resolved, optimized graph.
  // node_type must match its registered type ("MoE" or "QMoE"). expert_id is a zero-based
  // index local to that node, not a global expert index or kernel pointer.
  // counter_value is parsed in the classic locale and must be finite and non-negative.
  //
  // Omitted experts retain their current values (zero after registration). Blank lines,
  // comments, extra fields, duplicate expert records, unknown nodes, and invalid IDs or
  // values are rejected. A malformed record or stream read error leaves all counters unchanged.
  Status Load(std::istream& input);

  Status FinalizeInitialization();
  Status BeginRun() const;
  void EndRun() const;
  KernelPilot* GetKernelPilot(const OpKernel* kernel);
  // Commit the kernel's collected usage after a successful invocation.
  Status RecordUsage(const OpKernel* kernel);
  // During a Run, only this kernel may read its counters.
  Status GetCounters(const OpKernel* kernel, InlinedVector<double>& counters) const;
  Status GetExpertId(const OpKernel* kernel, int expert_id, size_t& global_expert_id) const;

  // One entry per registered (kernel, local expert index). expert_id is local to that kernel,
  // matching RegisterNode/GetExpertId, not a global counter index. Order is unspecified.
  // Call only while no Run is active.
  struct ExpertStat {
    const OpKernel* kernel;
    size_t expert_id;
    double popularity;
  };
  InlinedVector<ExpertStat> GetExpertStats() const;

  size_t TotalExpertCount() const noexcept { return counters_.size(); }

 private:
  using Key = std::pair<std::string, size_t>;
  struct ExpertRange {
    size_t begin;
    size_t count;
  };
  struct KernelState {
    KernelState(Key key, std::string node_type, ExpertRange range)
        : key(std::move(key)), node_type(std::move(node_type)), experts(range) {}
    Key key;
    std::string node_type;
    ExpertRange experts;
    KernelPilot pilot;
  };
  NodeHashMap<const OpKernel*, KernelState> kernels_;
  InlinedHashMap<std::pair<const OpKernel*, int>, size_t> expert_ids_;
  InlinedVector<double> counters_;
  double alpha_{0.9};
  double beta_{0.1};
  bool initialized_{false};
  // Reject overlapping runs once per Run, not once per kernel or expert.
  mutable std::atomic_flag run_active_ = ATOMIC_FLAG_INIT;
};

}  // namespace onnxruntime
