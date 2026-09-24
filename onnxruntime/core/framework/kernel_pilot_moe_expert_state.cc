// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_pilot_moe_expert_state.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

#include "core/common/json_utils.h"
#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"

namespace onnxruntime {

KernelPilotMoeExpertState::LoggingContext::LoggingContext(std::string request_id, const logging::Logger& logger)
    : request_id_(std::move(request_id)),
      logger_(logger),
      start_time_ns_(static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count())) {}

const std::string& KernelPilotMoeExpertState::LoggingContext::RequestId() const noexcept {
  return request_id_;
}

TimePoint KernelPilotMoeExpertState::LoggingContext::StartProfiling() const {
  return std::chrono::high_resolution_clock::now();
}

uint64_t KernelPilotMoeExpertState::LoggingContext::ProfilerStartTimeNs() const noexcept {
  return start_time_ns_;
}

void KernelPilotMoeExpertState::LoggingContext::RecordMoeRoutingEvent(
    const TimePoint&,
    const TimePoint&,
    std::string_view node_name,
    NodeIndex node_index,
    std::string_view node_type,
    std::string expert_ids_json,
    std::string router_weights_json,
    int64_t num_rows,
    int64_t top_k,
    int execution_device_id,
    int64_t,
    std::string_view) const {
  std::ostringstream event;
  event << "{\"request_id\":";
  common::WriteJsonString(event, request_id_);
  event << ",\"node_name\":";
  common::WriteJsonString(event, node_name);
  event << ",\"node_index\":" << node_index
        << ",\"node_type\":";
  common::WriteJsonString(event, node_type);
  event << ",\"expert_ids\":" << expert_ids_json
        << ",\"router_weights\":" << router_weights_json
        << ",\"num_rows\":" << num_rows
        << ",\"top_k\":" << top_k
        << ",\"execution_device_id\":" << execution_device_id
        << "}";
  LOGS(logger_, INFO) << "moe_routing " << event.str();
}

void KernelPilotMoeExpertState::LoggingContext::AddDeferredRecord(
    std::unique_ptr<KernelPilotMoeDeferredRecord> record) const {
  std::lock_guard<std::mutex> lock(deferred_records_mutex_);
  deferred_records_.push_back(std::move(record));
}

bool KernelPilotMoeExpertState::LoggingContext::TryReserveMoeRoutingRecord(size_t element_count) const {
  std::lock_guard<std::mutex> lock(deferred_records_mutex_);
  if (moe_routing_record_count_ >= kMaxMoeRoutingRecordsPerRun ||
      element_count > kMaxMoeRoutingElementsPerRun - moe_routing_element_count_) {
    ++dropped_moe_routing_record_count_;
    dropped_moe_routing_element_count_ += element_count;
    return false;
  }

  ++moe_routing_record_count_;
  moe_routing_element_count_ += element_count;
  return true;
}

void KernelPilotMoeExpertState::LoggingContext::LogMoeStatisticsTruncation() const {
  std::lock_guard<std::mutex> lock(deferred_records_mutex_);
  if (dropped_moe_routing_record_count_ == 0) {
    return;
  }

  LOGS(logger_, WARNING)
      << "moe_routing_truncated {\"dropped_records\":"
      << dropped_moe_routing_record_count_
      << ",\"dropped_routing_elements\":" << dropped_moe_routing_element_count_
      << ",\"max_records_per_run\":" << kMaxMoeRoutingRecordsPerRun
      << ",\"max_routing_elements_per_run\":" << kMaxMoeRoutingElementsPerRun
      << "}";
}

Status KernelPilotMoeExpertState::LoggingContext::FlushDeferredRecords() {
  InlinedVector<std::unique_ptr<KernelPilotMoeDeferredRecord>> records;
  {
    std::lock_guard<std::mutex> lock(deferred_records_mutex_);
    records = std::move(deferred_records_);
  }

  Status status = Status::OK();
  for (auto& record : records) {
    const std::string error_message = record->Emit();
    if (status.IsOK() && !error_message.empty()) {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_message);
    }
  }
  return status;
}

Status KernelPilotMoeExpertState::BeginLogging(std::string request_id, const logging::Logger& logger) {
  std::lock_guard<std::mutex> lock(logging_context_mutex_);
  ORT_RETURN_IF(logging_context_ != nullptr,
                "Concurrent Runs are not supported while MoE expert statistics logging is enabled.");
  logging_context_ = std::make_unique<LoggingContext>(std::move(request_id), logger);
  return Status::OK();
}

const IKernelPilotMoeLoggingContext* KernelPilotMoeExpertState::GetLoggingContext() const {
  std::lock_guard<std::mutex> lock(logging_context_mutex_);
  return logging_context_.get();
}

Status KernelPilotMoeExpertState::EndLogging() {
  std::unique_ptr<LoggingContext> logging_context;
  {
    std::lock_guard<std::mutex> lock(logging_context_mutex_);
    ORT_RETURN_IF_NOT(logging_context_, "MoE expert statistics logging is not active.");
    logging_context = std::move(logging_context_);
  }

  const Status status = logging_context->FlushDeferredRecords();
  logging_context->LogMoeStatisticsTruncation();
  return status;
}

Status KernelPilotMoeExpertState::SetCounterParameters(double alpha, double beta) {
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

Status KernelPilotMoeExpertState::RegisterNode(const OpKernel* kernel, std::string_view graph_scope, size_t node_index,
                                               std::string_view node_type, size_t expert_count) {
  ORT_RETURN_IF(initialized_, "MoE expert registration is closed.");
  ORT_RETURN_IF_NOT(kernel, "MoE expert registration requires a kernel.");
  ORT_RETURN_IF_NOT(node_type == "MoE" || node_type == "QMoE", "Unsupported expert counter node type: ", node_type);
  ORT_RETURN_IF(expert_count == 0 || expert_count > static_cast<size_t>(std::numeric_limits<int>::max()),
                "Expert count must be positive and fit in an int.");
  const Key key{std::string(graph_scope), node_index};
  ORT_RETURN_IF(kernels_.contains(kernel), "Duplicate MoE counter kernel: ", graph_scope, " ", node_index);
  for (const auto& [existing_kernel, state] : kernels_) {
    ORT_RETURN_IF(state.key == key, "Duplicate MoE counter node: ", graph_scope, " ", node_index);
  }
  const ExpertRange range{counters_.size(), expert_count};
  const size_t total_expert_count = SafeInt<size_t>(range.begin) + expert_count;
  counters_.reserve(total_expert_count);
  expert_ids_.reserve(total_expert_count);
  for (size_t expert = 0; expert < expert_count; ++expert) {
    expert_ids_.emplace(std::make_pair(kernel, static_cast<int>(expert)), counters_.size());
    counters_.push_back(0.0);
  }
  auto [entry, inserted] = kernels_.try_emplace(kernel, key, range);
  ORT_ENFORCE(inserted);
  ORT_RETURN_IF_ERROR(entry->second.pilot.Moe().BeginInvocation(expert_count));
  return Status::OK();
}

Status KernelPilotMoeExpertState::Load(std::istream& input) {
  ORT_RETURN_IF(initialized_, "MoE expert initial state cannot change after initialization.");
  std::string line;
  ORT_RETURN_IF_NOT(std::getline(input, line), "Missing initial expert counter state header.");
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  ORT_RETURN_IF_NOT(line == "moe_expert_state 1",
                    "Expected initial counter state header: moe_expert_state 1");

  // Validate into a flat copy so a malformed file cannot partially overwrite the state.
  // Build a local graph-identity index from kernels_'s stored keys, just for resolving this
  // file's records; KernelPilotMoeExpertState itself only ever looks kernels up by pointer.
  std::map<Key, const OpKernel*> nodes;
  for (const auto& [kernel, state] : kernels_) {
    nodes.emplace(state.key, kernel);
  }
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
    const auto node = nodes.find({scope, static_cast<size_t>(node_index)});
    ORT_RETURN_IF(node == nodes.end(), "Unknown MoE counter node at line ", line_number);
    const auto& kernel_state = kernels_.at(node->second);
    ORT_RETURN_IF_NOT(node->second->Node().OpType() == type &&
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

Status KernelPilotMoeExpertState::FinalizeInitialization() {
  ORT_RETURN_IF(initialized_, "MoE expert state is already initialized.");
  initialized_ = true;
  return Status::OK();
}

KernelPilot* KernelPilotMoeExpertState::GetKernelPilot(const OpKernel* kernel) {
  const auto node = kernels_.find(kernel);
  return node != kernels_.end() ? &node->second.pilot : nullptr;
}

Status KernelPilotMoeExpertState::RecordUsage(const OpKernel* kernel) {
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

Status KernelPilotMoeExpertState::GetCounters(const OpKernel* kernel, InlinedVector<double>& counters) const {
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

Status KernelPilotMoeExpertState::GetExpertId(const OpKernel* kernel, int expert_id, size_t& global_expert_id) const {
  const auto expert = expert_ids_.find({kernel, expert_id});
  ORT_RETURN_IF(expert == expert_ids_.end(), "Unknown MoE kernel/expert pair: ", expert_id);
  global_expert_id = expert->second;
  return Status::OK();
}

InlinedVector<KernelPilotMoeExpertState::ExpertStat> KernelPilotMoeExpertState::GetExpertStats() const {
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
