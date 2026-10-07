// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_pilot_moe_expert_state.h"

#include <algorithm>
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
#include "core/graph/constants.h"

namespace onnxruntime {

Status KernelPilotMoeExpertState::BeginRun(std::string request_id, const logging::Logger* logger) {
  std::lock_guard<std::mutex> lock(run_mutex_);
  ORT_RETURN_IF(run_active_, "Concurrent Runs are not supported while MoE expert tracking is enabled.");
  run_active_ = true;
  logging_request_id_ = std::move(request_id);
  logging_logger_ = logger;
  logging_record_count_.store(0, std::memory_order_relaxed);
  return Status::OK();
}

Status KernelPilotMoeExpertState::EndRun(bool run_succeeded) {
  std::lock_guard<std::mutex> lock(run_mutex_);
  ORT_RETURN_IF_NOT(run_active_, "MoE expert tracking is not active.");
  Status status = Status::OK();
  if (run_succeeded && cpu_offload_enabled_) {
    status = ScheduleSwaps();
  }
  logging_request_id_.clear();
  logging_logger_ = nullptr;
  run_active_ = false;
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

Status KernelPilotMoeExpertState::SetCpuOffloadExpertCount(size_t cpu_offload_expert_count) {
  ORT_RETURN_IF(initialized_, "MoE CPU offload expert count cannot change after initialization.");
  cpu_offload_expert_count_ = cpu_offload_expert_count;
  cpu_offload_enabled_ = cpu_offload_expert_count > 0;
  return Status::OK();
}

Status KernelPilotMoeExpertState::SetSwapEpsilon(double epsilon) {
  ORT_RETURN_IF_NOT(std::isfinite(epsilon) && epsilon >= 0.0,
                    "MoE expert swap epsilon must be finite and non-negative.");
  ORT_RETURN_IF(initialized_ || !kernels_.empty(),
                "MoE expert swap epsilon cannot change after node registration.");
  swap_epsilon_ = epsilon;
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
  auto [entry, inserted] = kernels_.try_emplace(kernel, key, range, *this, kernel);
  ORT_ENFORCE(inserted);
  ORT_RETURN_IF_ERROR(entry->second.pilot.Moe().BeginInvocation(expert_count));
  entry->second.pilot.FinishRegistration();
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

  if (cpu_offload_enabled_) {
    struct Candidate {
      KernelState* state;
      int expert_id;
      double counter;
    };

    InlinedVector<KernelState*> cuda_kernels;
    size_t cuda_eligible_expert_count = 0;
    for (auto& [kernel, state] : kernels_) {
      const auto* input_type = kernel->Node().InputDefs().empty()
                                   ? nullptr
                                   : kernel->Node().InputDefs()[0]->TypeAsProto();
      const int32_t element_type =
          input_type != nullptr && input_type->has_tensor_type()
              ? input_type->tensor_type().elem_type()
              : ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
      if (kernel->Node().GetExecutionProviderType() == kCudaExecutionProvider &&
          kernel->Node().Domain() == kMSDomain && kernel->Node().OpType() == "MoE" &&
          (element_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
           element_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16)) {
        cuda_kernels.push_back(&state);
        cuda_eligible_expert_count += state.experts.count;
      }
    }
    ORT_RETURN_IF(cpu_offload_expert_count_ > cuda_eligible_expert_count,
                  "session.moe_cpu_offload_experts is ", cpu_offload_expert_count_,
                  ", but CUDA FP16/BF16 MoE nodes contain only ", cuda_eligible_expert_count, " experts.");
    const size_t cuda_expert_count = cuda_eligible_expert_count - cpu_offload_expert_count_;

    std::sort(cuda_kernels.begin(), cuda_kernels.end(), [](const KernelState* lhs, const KernelState* rhs) {
      return lhs->key < rhs->key;
    });

    bool all_zero = true;
    for (const auto* state : cuda_kernels) {
      for (size_t expert = 0; expert < state->experts.count; ++expert) {
        all_zero = all_zero && counters_[state->experts.begin + expert] == 0.0;
      }
    }

    if (all_zero) {
      size_t selected = 0;
      for (size_t expert = 0; selected < cuda_expert_count; ++expert) {
        bool made_progress = false;
        for (auto* state : cuda_kernels) {
          if (expert < state->experts.count && selected < cuda_expert_count) {
            state->cuda_experts.push_back(static_cast<int>(expert));
            ++selected;
            made_progress = true;
          }
        }
        ORT_ENFORCE(made_progress || selected == cuda_expert_count);
      }
    } else {
      InlinedVector<Candidate> candidates;
      candidates.reserve(cuda_eligible_expert_count);
      for (auto* state : cuda_kernels) {
        for (size_t expert = 0; expert < state->experts.count; ++expert) {
          candidates.push_back(
              {state, static_cast<int>(expert), counters_[state->experts.begin + expert]});
        }
      }
      std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.counter != rhs.counter) {
          return lhs.counter > rhs.counter;
        }
        if (lhs.expert_id != rhs.expert_id) {
          return lhs.expert_id < rhs.expert_id;
        }
        return lhs.state->key < rhs.state->key;
      });
      for (size_t i = 0; i < cuda_expert_count; ++i) {
        candidates[i].state->cuda_experts.push_back(candidates[i].expert_id);
      }
      for (auto* state : cuda_kernels) {
        std::sort(state->cuda_experts.begin(), state->cuda_experts.end());
      }
    }
  }

  for (auto& [kernel, state] : kernels_) {
    ORT_UNUSED_PARAMETER(kernel);
    state.pilot.SetMoeCudaExperts(state.cuda_experts);
  }

  initialized_ = true;
  return Status::OK();
}

KernelPilot* KernelPilotMoeExpertState::GetKernelPilot(const OpKernel* kernel) {
  const auto node = kernels_.find(kernel);
  return node != kernels_.end() ? &node->second.pilot : nullptr;
}

Status KernelPilotMoeExpertState::ScheduleSwaps() {
  constexpr size_t kMaxInFlightSwapsPerDevice = 4;
  InlinedHashMap<int, size_t> pending_by_device;
  InlinedVector<KernelState*> ordered_states;
  ordered_states.reserve(kernels_.size());
  for (auto& [kernel, state] : kernels_) {
    ORT_UNUSED_PARAMETER(kernel);
    auto* cache = state.pilot.GetMoeExpertCache();
    if (cache != nullptr) {
      ORT_RETURN_IF_ERROR(cache->ReclaimCompletedSwap());
      if (cache->HasPendingSwap()) {
        ++pending_by_device[cache->DeviceId()];
      }
      ordered_states.push_back(&state);
    }
  }
  std::sort(ordered_states.begin(), ordered_states.end(),
            [](const KernelState* lhs, const KernelState* rhs) { return lhs->key < rhs->key; });

  for (auto* state : ordered_states) {
    auto* cache = state->pilot.GetMoeExpertCache();
    const int device_id = cache->DeviceId();
    if (cache->HasPendingSwap() ||
        pending_by_device[device_id] >= kMaxInFlightSwapsPerDevice ||
        state->cuda_experts.empty() ||
        state->cuda_experts.size() == state->experts.count) {
      continue;
    }

    int coldest_cuda_expert = state->cuda_experts.front();
    double cuda_min = counters_[state->experts.begin + static_cast<size_t>(coldest_cuda_expert)];
    for (int expert : state->cuda_experts) {
      const double counter = counters_[state->experts.begin + static_cast<size_t>(expert)];
      if (counter < cuda_min || (counter == cuda_min && expert < coldest_cuda_expert)) {
        coldest_cuda_expert = expert;
        cuda_min = counter;
      }
    }

    int hottest_cpu_expert = -1;
    double cpu_max = 0.0;
    for (size_t expert = 0; expert < state->experts.count; ++expert) {
      const int expert_id = static_cast<int>(expert);
      if (std::find(state->cuda_experts.begin(), state->cuda_experts.end(), expert_id) !=
          state->cuda_experts.end()) {
        continue;
      }
      const double counter = counters_[state->experts.begin + expert];
      if (hottest_cpu_expert < 0 || counter > cpu_max ||
          (counter == cpu_max && expert_id < hottest_cpu_expert)) {
        hottest_cpu_expert = expert_id;
        cpu_max = counter;
      }
    }
    ORT_ENFORCE(hottest_cpu_expert >= 0);
    if (cpu_max > (1.0 + swap_epsilon_) * cuda_min) {
      ORT_RETURN_IF_ERROR(cache->StartSwap(coldest_cuda_expert, hottest_cpu_expert));
      ++pending_by_device[device_id];
    }
  }
  return Status::OK();
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

  if (logging_logger_ != nullptr &&
      logging_logger_->OutputIsEnabled(logging::Severity::kINFO, logging::DataType::SYSTEM)) {
    // Reserve across parallel nodes before formatting. Counters remain independent of the log budget.
    if (logging_record_count_.load(std::memory_order_relaxed) > kMaxCounterLogRecordsPerRun) {
      return Status::OK();
    }
    const size_t record_index = logging_record_count_.fetch_add(1, std::memory_order_relaxed);
    if (record_index >= kMaxCounterLogRecordsPerRun) {
      if (record_index == kMaxCounterLogRecordsPerRun) {
        std::ostringstream summary;
        summary.imbue(std::locale::classic());
        summary << "{\"request_id\":";
        common::WriteJsonString(summary, logging_request_id_);
        summary << ",\"max_records\":" << kMaxCounterLogRecordsPerRun << "}";
        LOGS(*logging_logger_, WARNING) << "moe_expert_counters_truncated " << summary.str();
      }
      return Status::OK();
    }
    std::ostringstream event;
    event.imbue(std::locale::classic());
    event << "{\"request_id\":";
    common::WriteJsonString(event, logging_request_id_);
    event << ",\"graph_scope\":";
    common::WriteJsonString(event, node->second.key.first);
    event << ",\"node_name\":";
    common::WriteJsonString(event, kernel->Node().Name());
    event << ",\"node_index\":" << node->second.key.second
          << ",\"node_type\":";
    common::WriteJsonString(event, kernel->Node().OpType());
    event << ",\"selected_experts\":[";
    for (size_t i = 0; i < used_expert_ids.size(); ++i) {
      if (i != 0) {
        event << ",";
      }
      event << used_expert_ids[i];
    }
    event << "],\"counters\":[";
    for (size_t i = 0; i < counters.size(); ++i) {
      if (i != 0) {
        event << ",";
      }
      event << std::setprecision(std::numeric_limits<double>::max_digits10) << counters[i];
    }
    event << "]}";
    LOGS(*logging_logger_, INFO) << "moe_expert_counters " << event.str();
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
