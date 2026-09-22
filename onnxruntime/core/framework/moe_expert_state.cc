// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/moe_expert_state.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <tuple>

namespace onnxruntime {

Status MoeExpertState::RegisterNode(std::string_view graph_scope, size_t node_index,
                                    std::string_view node_type, size_t expert_count) {
  ORT_RETURN_IF_NOT(node_type == "MoE" || node_type == "QMoE", "Unsupported expert counter node type: ", node_type);
  ORT_RETURN_IF(expert_count == 0, "Expert count must be positive.");
  std::lock_guard<std::mutex> lock(mutex_);
  const bool inserted = nodes_.emplace(
                                  Key{std::string(graph_scope), node_index},
                                  NodeCounters{std::string(node_type), InlinedVector<double>(expert_count, 0.0)})
                            .second;
  ORT_RETURN_IF_NOT(inserted, "Duplicate MoE counter node: ", graph_scope, " ", node_index);
  return Status::OK();
}

Status MoeExpertState::Load(std::istream& input) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string line;
  ORT_RETURN_IF_NOT(std::getline(input, line), "Missing initial expert counter state header.");
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  ORT_RETURN_IF_NOT(line == "moe_expert_state 1",
                    "Expected initial counter state header: moe_expert_state 1");

  // Validate into a copy so a malformed file cannot partially overwrite the state.
  auto loaded = nodes_;
  std::set<std::tuple<std::string, size_t, size_t>> seen;
  size_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    std::istringstream record(line);
    record.imbue(std::locale::classic());
    std::string scope, type;
    int64_t node_index = -1, expert_id = -1;
    double value = 0;
    ORT_RETURN_IF_NOT(record >> std::quoted(scope) >> node_index >> type >> expert_id >> value,
                      "Malformed expert counter record at line ", line_number);
    record >> std::ws;
    ORT_RETURN_IF_NOT(record.eof() && node_index >= 0 && expert_id >= 0 &&
                          static_cast<uint64_t>(node_index) <= std::numeric_limits<size_t>::max() &&
                          static_cast<uint64_t>(expert_id) <= std::numeric_limits<size_t>::max() &&
                          std::isfinite(value) && value >= 0,
                      "Invalid expert counter record at line ", line_number);
    const auto node = loaded.find({scope, static_cast<size_t>(node_index)});
    ORT_RETURN_IF(node == loaded.end(), "Unknown MoE counter node at line ", line_number);
    ORT_RETURN_IF_NOT(node->second.node_type == type &&
                          static_cast<size_t>(expert_id) < node->second.counters.size(),
                      "MoE counter type or expert index mismatch at line ", line_number);
    ORT_RETURN_IF_NOT(seen.emplace(scope, static_cast<size_t>(node_index), static_cast<size_t>(expert_id)).second,
                      "Duplicate expert counter at line ", line_number);
    node->second.counters[static_cast<size_t>(expert_id)] = value;
  }
  ORT_RETURN_IF(input.bad() || !input.eof(), "Failed to read initial expert counter state.");
  nodes_ = std::move(loaded);
  return Status::OK();
}

Status MoeExpertState::RecordUsage(std::string_view graph_scope, size_t node_index,
                                   gsl::span<const int> expert_ids) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto node = nodes_.find({std::string(graph_scope), node_index});
  ORT_RETURN_IF(node == nodes_.end(), "Unknown MoE counter node: ", graph_scope, " ", node_index);
  auto& counters = node->second.counters;
  InlinedHashSet<int> used;
  for (int expert : expert_ids) {
    ORT_RETURN_IF(expert < 0 || static_cast<size_t>(expert) >= counters.size(),
                  "MoE counter expert index out of range: ", expert);
    used.insert(expert);
  }
  for (int expert : used) {
    ORT_RETURN_IF(counters[expert] + 1.0 == counters[expert] || !std::isfinite(counters[expert] + 1.0),
                  "MoE expert counter cannot be incremented: ", expert);
  }
  for (int expert : used) {
    counters[expert] += 1.0;
  }
  return Status::OK();
}

Status MoeExpertState::GetCounters(std::string_view graph_scope, size_t node_index,
                                   InlinedVector<double>& counters) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto node = nodes_.find({std::string(graph_scope), node_index});
  ORT_RETURN_IF(node == nodes_.end(), "Unknown MoE counter node: ", graph_scope, " ", node_index);
  counters = node->second.counters;
  return Status::OK();
}

MoeExpertState::Snapshot MoeExpertState::GetSnapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return nodes_;
}

}  // namespace onnxruntime
