// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <istream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <gsl/gsl>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {

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

  Status RegisterNode(std::string_view graph_scope, size_t node_index,
                      std::string_view node_type, size_t expert_count);
  Status Load(std::istream& input);
  Status RecordUsage(std::string_view graph_scope, size_t node_index, gsl::span<const int> expert_ids);
  Status GetCounters(std::string_view graph_scope, size_t node_index, InlinedVector<double>& counters) const;
  Snapshot GetSnapshot() const;

 private:
  mutable std::mutex mutex_;
  Snapshot nodes_;
};

}  // namespace onnxruntime
