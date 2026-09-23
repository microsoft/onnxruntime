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

#include "core/common/safeint.h"

namespace onnxruntime {

Status MoeExpertState::SetCounterParameters(double alpha, double beta) {
  ORT_RETURN_IF_NOT(std::isfinite(alpha) && alpha >= 0.0,
                    "MoE expert counter alpha must be finite and non-negative.");
  ORT_RETURN_IF_NOT(std::isfinite(beta) && beta >= 0.0,
                    "MoE expert counter beta must be finite and non-negative.");
  ORT_RETURN_IF_NOT(alpha + beta <= 1.0, "MoE expert counter alpha + beta must be at most 1.");
  ORT_RETURN_IF(initialized_ || !kernels_.empty(),
                "MoE expert counter parameters cannot change after node registration.");
  alpha_ = alpha;
  beta_ = beta;
  return Status::OK();
}

Status MoeExpertState::RegisterNode(const OpKernel* kernel, std::string_view graph_scope, size_t node_index,
                                    std::string_view node_type, size_t expert_count) {
  ORT_RETURN_IF(initialized_, "MoE expert registration is closed.");
  ORT_RETURN_IF_NOT(kernel, "MoE expert registration requires a kernel.");
  ORT_RETURN_IF_NOT(node_type == "MoE" || node_type == "QMoE", "Unsupported expert counter node type: ", node_type);
  ORT_RETURN_IF(expert_count == 0 || expert_count > static_cast<size_t>(std::numeric_limits<int>::max()),
                "Expert count must be positive and fit in an int.");
  const Key key{std::string(graph_scope), node_index};
  ORT_RETURN_IF(nodes_.find(key) != nodes_.end(), "Duplicate MoE counter node: ", graph_scope, " ", node_index);
  ORT_RETURN_IF(kernels_.contains(kernel), "Duplicate MoE counter kernel: ", graph_scope, " ", node_index);
  const ExpertRange range{counters_.size(), expert_count};
  const size_t total_expert_count = SafeInt<size_t>(range.begin) + expert_count;
  counters_.reserve(total_expert_count);
  expert_ids_.reserve(total_expert_count);
  for (size_t expert = 0; expert < expert_count; ++expert) {
    expert_ids_.emplace(std::make_pair(kernel, static_cast<int>(expert)), counters_.size());
    counters_.push_back(0.0);
  }
  nodes_.emplace(key, kernel);
  auto [entry, inserted] = kernels_.try_emplace(kernel, std::string(node_type), range);
  ORT_ENFORCE(inserted);
  ORT_RETURN_IF_ERROR(entry->second.pilot.Moe().BeginInvocation(expert_count));
  return Status::OK();
}

Status MoeExpertState::Load(std::istream& input) {
  ORT_RETURN_IF(initialized_, "MoE expert initial state cannot change after initialization.");
  std::string line;
  ORT_RETURN_IF_NOT(std::getline(input, line), "Missing initial expert counter state header.");
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  ORT_RETURN_IF_NOT(line == "moe_expert_state 1",
                    "Expected initial counter state header: moe_expert_state 1");

  // Validate into a flat copy so a malformed file cannot partially overwrite the state.
  // nodes_ resolves graph identity to a kernel; kernels_ holds that kernel's (node_type, range).
  auto loaded_counters = counters_;
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
    const auto node = nodes_.find({scope, static_cast<size_t>(node_index)});
    ORT_RETURN_IF(node == nodes_.end(), "Unknown MoE counter node at line ", line_number);
    const auto& kernel_state = kernels_.at(node->second);
    ORT_RETURN_IF_NOT(kernel_state.node_type == type &&
                          static_cast<size_t>(expert_id) < kernel_state.experts.count,
                      "MoE counter type or expert index mismatch at line ", line_number);
    ORT_RETURN_IF_NOT(seen.emplace(scope, static_cast<size_t>(node_index), static_cast<size_t>(expert_id)).second,
                      "Duplicate expert counter at line ", line_number);
    loaded_counters[kernel_state.experts.begin + static_cast<size_t>(expert_id)] = value;
  }
  ORT_RETURN_IF(input.bad() || !input.eof(), "Failed to read initial expert counter state.");
  counters_ = std::move(loaded_counters);
  return Status::OK();
}

Status MoeExpertState::FinalizeInitialization() {
  ORT_RETURN_IF(initialized_, "MoE expert state is already initialized.");
  initialized_ = true;
  // nodes_ only resolves graph-scope-keyed Load() records to their kernel; Load() must run
  // before this point, so the mapping is no longer needed once initialization is finalized.
  nodes_.clear();
  return Status::OK();
}

Status MoeExpertState::BeginRun() const {
  ORT_RETURN_IF_NOT(initialized_, "MoE expert state is not initialized.");
  ORT_RETURN_IF(run_active_.test_and_set(std::memory_order_acquire),
                "MoE expert counting does not support simultaneous Run calls on the same session.");
  return Status::OK();
}

void MoeExpertState::EndRun() const {
  run_active_.clear(std::memory_order_release);
}

KernelPilot* MoeExpertState::GetKernelPilot(const OpKernel* kernel) {
  const auto node = kernels_.find(kernel);
  return node != kernels_.end() ? &node->second.pilot : nullptr;
}

Status MoeExpertState::RecordUsage(const OpKernel* kernel) {
  ORT_RETURN_IF_NOT(initialized_, "MoE expert state is not initialized.");
  const auto node = kernels_.find(kernel);
  ORT_RETURN_IF(node == kernels_.end(), "Unknown MoE counter kernel.");
  const auto range = node->second.experts;
  const auto& usage = node->second.pilot.Moe();
  ORT_RETURN_IF_NOT(usage.ExpertCount() == range.count, "MoE kernel usage expert count does not match registration.");
  gsl::span<const int> used_expert_ids;
  ORT_RETURN_IF_ERROR(usage.GetSelectedExperts(used_expert_ids));
  auto counters = gsl::make_span(counters_).subspan(range.begin, range.count);
  for (auto& counter : counters) {
    counter *= alpha_;
  }
  for (int expert : used_expert_ids) {
    counters_[expert_ids_.at({kernel, expert})] += beta_;
  }
  return Status::OK();
}

Status MoeExpertState::GetCounters(const OpKernel* kernel, InlinedVector<double>& counters) const {
  const auto node = kernels_.find(kernel);
  ORT_RETURN_IF(node == kernels_.end(), "Unknown MoE counter kernel.");
  const auto range = node->second.experts;
  counters.clear();
  counters.reserve(range.count);
  for (size_t expert = 0; expert < range.count; ++expert) {
    counters.push_back(counters_[range.begin + expert]);
  }
  return Status::OK();
}

Status MoeExpertState::GetExpertId(const OpKernel* kernel, int expert_id, size_t& global_expert_id) const {
  const auto expert = expert_ids_.find({kernel, expert_id});
  ORT_RETURN_IF(expert == expert_ids_.end(), "Unknown MoE kernel/expert pair: ", expert_id);
  global_expert_id = expert->second;
  return Status::OK();
}

InlinedVector<MoeExpertState::ExpertStat> MoeExpertState::GetExpertStats() const {
  InlinedVector<ExpertStat> stats;
  stats.reserve(counters_.size());
  for (const auto& [kernel, state] : kernels_) {
    for (size_t expert = 0; expert < state.experts.count; ++expert) {
      stats.push_back({kernel, expert, counters_[state.experts.begin + expert]});
    }
  }
  return stats;
}

}  // namespace onnxruntime
