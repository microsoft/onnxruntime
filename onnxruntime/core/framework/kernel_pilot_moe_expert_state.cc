// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_pilot_moe_expert_state.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
#include <sstream>
#include <utility>

#include "nlohmann/json.hpp"
#include "core/common/json_utils.h"
#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "core/graph/constants.h"
#include "core/platform/path_lib.h"

namespace onnxruntime {

KernelPilotMoeExpertState::~KernelPilotMoeExpertState() {
  if (!popularity_output_file_.empty()) {
    const Status status = SavePopularity();
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << status.ErrorMessage();
    }
  }
}

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
  auto finish = gsl::finally([&]() {
    deferred_usage_.clear();
    logging_request_id_.clear();
    logging_logger_ = nullptr;
    run_active_ = false;
  });
  const auto complete_usage = [](KernelPilot& pilot) {
    Status completion;
    ORT_TRY {
      completion = pilot.CompleteMoeUsage();
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        completion = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "MoE usage completion failed: ", ex.what());
      });
    }
    ORT_CATCH(...) {
      completion = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Unknown exception during MoE usage completion.");
    }
    return completion;
  };
  Status status = Status::OK();
  for (auto* pilot : deferred_usage_) {
    const Status completion = complete_usage(*pilot);
    if (status.IsOK() && !completion.IsOK()) {
      status = completion;
    }
  }
  deferred_usage_.clear();
  for (auto& [kernel, state] : kernels_) {
    ORT_UNUSED_PARAMETER(kernel);
    const Status completion = complete_usage(state.pilot);
    if (status.IsOK() && !completion.IsOK()) {
      status = completion;
    }
  }
  if (run_succeeded && status.IsOK() && cpu_offload_enabled_ &&
      cpu_offload_strategy_ != CpuOffloadStrategy::Basic) {
    status = ScheduleSwaps();
  }
  return status;
}

Status KernelPilotMoeExpertState::SetCounterAlpha(double alpha) {
  ORT_RETURN_IF_NOT(std::isfinite(alpha) && alpha >= 0.0 && alpha <= 1.0,
                    "MoE expert counter alpha must be finite and in [0, 1].");
  ORT_RETURN_IF(initialized_ || !kernels_.empty(),
                "MoE expert counter parameters cannot change after node registration.");
  alpha_ = alpha;
  return Status::OK();
}

Status KernelPilotMoeExpertState::SetCounterProbabilityWeight(double probability_weight) {
  ORT_RETURN_IF_NOT(std::isfinite(probability_weight) &&
                        probability_weight >= 0.0 && probability_weight <= 1.0,
                    "MoE expert counter probability weight must be finite and in [0, 1].");
  ORT_RETURN_IF(initialized_ || !kernels_.empty(),
                "MoE expert counter probability weight cannot change after node registration.");
  counter_probability_weight_ = probability_weight;
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

Status KernelPilotMoeExpertState::SetCpuOffloadStrategy(std::string_view strategy) {
  ORT_RETURN_IF(initialized_ || !kernels_.empty(),
                "MoE CPU offload strategy cannot change after node registration.");
  ORT_RETURN_IF_NOT(strategy == "adaptive" || strategy == "basic" || strategy == "layered",
                    "MoE CPU offload strategy must be adaptive, basic or layered. Received: ", strategy);
  cpu_offload_strategy_ = strategy == "basic"     ? CpuOffloadStrategy::Basic
                          : strategy == "layered" ? CpuOffloadStrategy::Layered
                                                  : CpuOffloadStrategy::Adaptive;
  return Status::OK();
}

Status KernelPilotMoeExpertState::SetCpuOffloadGpuRatio(double gpu_ratio) {
  ORT_RETURN_IF(initialized_ || !kernels_.empty(),
                "MoE CPU offload GPU ratio cannot change after node registration.");
  ORT_RETURN_IF_NOT(std::isfinite(gpu_ratio) && gpu_ratio > 0.0 && gpu_ratio <= 1.0,
                    "MoE CPU offload GPU ratio must be finite and in (0, 1].");
  cpu_offload_gpu_ratio_ = gpu_ratio;
  return Status::OK();
}

Status KernelPilotMoeExpertState::SetMaxInFlightSwapsPerDevice(size_t max_in_flight_swaps_per_device) {
  ORT_RETURN_IF(initialized_ || !kernels_.empty(),
                "MoE maximum in-flight swaps per device cannot change after node registration.");
  max_in_flight_swaps_per_device_ = max_in_flight_swaps_per_device;
  return Status::OK();
}

Status KernelPilotMoeExpertState::SetCpuOffloadLayerOrder(std::string_view layer_order) {
  ORT_RETURN_IF(initialized_ || !kernels_.empty(),
                "MoE CPU offload layer order cannot change after node registration.");
  ORT_RETURN_IF_NOT(layer_order == "first" || layer_order == "last" || layer_order == "middle",
                    "MoE CPU offload layer order must be first, last or middle. Received: ", layer_order);
  cpu_offload_layer_order_ = layer_order == "last"     ? CpuOffloadLayerOrder::Last
                             : layer_order == "middle" ? CpuOffloadLayerOrder::Middle
                                                       : CpuOffloadLayerOrder::First;
  return Status::OK();
}

Status KernelPilotMoeExpertState::SetPopularityOutputFile(std::string output_file) {
  ORT_RETURN_IF(output_file.empty(), "MoE expert popularity output file must not be empty.");
  ORT_RETURN_IF(!popularity_output_file_.empty(), "MoE expert popularity output file is already configured.");
  popularity_output_file_ = std::move(output_file);
  return Status::OK();
}

Status KernelPilotMoeExpertState::RegisterNode(const OpKernel* kernel, std::string_view graph_scope, size_t node_index,
                                               std::string_view node_type, size_t expert_count, size_t topological_order) {
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
  auto [entry, inserted] = kernels_.try_emplace(kernel, key, range, *this, kernel, topological_order);
  ORT_ENFORCE(inserted);
  ORT_RETURN_IF_ERROR(entry->second.pilot.Moe().BeginInvocation(expert_count));
  entry->second.pilot.FinishRegistration();
  return Status::OK();
}

Status KernelPilotMoeExpertState::Load(std::istream& input) {
  ORT_RETURN_IF(initialized_, "MoE expert initial state cannot change after initialization.");
  const auto document = nlohmann::json::parse(input, nullptr, false);
  ORT_RETURN_IF(document.is_discarded() || !document.is_object(),
                "MoE expert initial state must be a JSON object.");
  const auto alpha = document.find("alpha");
  const auto epsilon = document.find("epsilon");
  const auto experts = document.find("experts");
  ORT_RETURN_IF_NOT(alpha != document.end() && alpha->is_number() &&
                        epsilon != document.end() && epsilon->is_number() &&
                        experts != document.end() && experts->is_array(),
                    "MoE expert initial state requires numeric alpha and epsilon fields and an experts array.");
  const double loaded_alpha = alpha->get<double>();
  const double loaded_epsilon = epsilon->get<double>();
  ORT_RETURN_IF_NOT(std::isfinite(loaded_alpha) && loaded_alpha >= 0.0 && loaded_alpha <= 1.0,
                    "MoE expert initial state alpha must be finite and in [0, 1].");
  ORT_RETURN_IF_NOT(std::isfinite(loaded_epsilon) && loaded_epsilon >= 0.0,
                    "MoE expert initial state epsilon must be finite and non-negative.");

  InlinedVector<const KernelState*> states;
  states.reserve(kernels_.size());
  for (const auto& [kernel, state] : kernels_) {
    ORT_UNUSED_PARAMETER(kernel);
    states.push_back(&state);
  }
  std::sort(states.begin(), states.end(),
            [](const KernelState* lhs, const KernelState* rhs) { return lhs->key < rhs->key; });

  auto loaded_counters = counters_;
  InlinedVector<bool> seen(counters_.size(), false);
  for (size_t record_index = 0; record_index < experts->size(); ++record_index) {
    const auto& record = (*experts)[record_index];
    ORT_RETURN_IF_NOT(record.is_object(), "MoE expert record ", record_index, " must be an object.");
    const auto moe = record.find("moe");
    const auto expert = record.find("expert");
    const auto popularity = record.find("popularity");
    ORT_RETURN_IF_NOT(moe != record.end() && moe->is_number_unsigned() &&
                          expert != record.end() && expert->is_number_unsigned() &&
                          popularity != record.end() && popularity->is_number(),
                      "MoE expert record ", record_index,
                      " requires non-negative integer moe and expert fields and numeric popularity.");
    const uint64_t moe_index = moe->get<uint64_t>();
    const uint64_t expert_index = expert->get<uint64_t>();
    const double value = popularity->get<double>();
    ORT_RETURN_IF_NOT(moe_index < states.size() &&
                          expert_index < states[static_cast<size_t>(moe_index)]->experts.count &&
                          std::isfinite(value) && value >= 0.0,
                      "MoE expert record ", record_index, " is out of range or has invalid popularity.");
    const size_t counter_index =
        states[static_cast<size_t>(moe_index)]->experts.begin + static_cast<size_t>(expert_index);
    ORT_RETURN_IF(seen[counter_index], "Duplicate MoE expert record ", record_index, ".");
    seen[counter_index] = true;
    loaded_counters[counter_index] = value;
  }

  alpha_ = loaded_alpha;
  swap_epsilon_ = loaded_epsilon;
  counters_ = std::move(loaded_counters);
  return Status::OK();
}

bool KernelPilotMoeExpertState::IsCpuOffloadEligible(const Node& node) {
  if (node.GetExecutionProviderType() != kCudaExecutionProvider || node.Domain() != kMSDomain ||
      (node.OpType() != "MoE" && node.OpType() != "QMoE") || node.InputDefs().empty()) {
    return false;
  }
  const auto* input_type = node.InputDefs()[0]->TypeAsProto();
  if (input_type == nullptr || !input_type->has_tensor_type()) {
    return false;
  }
  const int32_t element_type = input_type->tensor_type().elem_type();
  if (element_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
      element_type != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
    return false;
  }
  if (node.OpType() == "MoE") {
    return true;
  }
  const auto& attributes = node.GetAttributes();
  const auto get_int_attribute = [&](std::string_view name, int64_t default_value) {
    const auto attribute = attributes.find(std::string(name));
    return attribute == attributes.end() ? default_value : attribute->second.i();
  };
  const auto quant_type = attributes.find("quant_type");
  const int64_t bits = get_int_attribute("expert_weight_bits", 0);
  return (quant_type == attributes.end() || quant_type->second.s() == "int") &&
         (bits == 4 || bits == 8) &&
         get_int_attribute("fc1_expert_weight_bits", bits) == bits &&
         get_int_attribute("fc2_expert_weight_bits", bits) == bits &&
         get_int_attribute("weights_prepacked", -1) == 0;
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
      const auto& node = kernel->Node();
      ORT_RETURN_IF(cpu_offload_strategy_ != CpuOffloadStrategy::Adaptive && node.OpType() == "MoE" &&
                        node.GetExecutionProviderType() == kCudaExecutionProvider,
                    "The ", cpu_offload_strategy_ == CpuOffloadStrategy::Basic ? "basic" : "layered",
                    " CPU offload strategy supports QMoE only, not CUDA MoE.");
      if (IsCpuOffloadEligible(node)) {
        cuda_kernels.push_back(&state);
        cuda_eligible_expert_count += state.experts.count;
      }
    }
    ORT_RETURN_IF(cpu_offload_expert_count_ > cuda_eligible_expert_count,
                  "session.moe_cpu_offload_experts is ", cpu_offload_expert_count_,
                  ", but supported CUDA FP16/BF16 MoE/QMoE nodes contain only ",
                  cuda_eligible_expert_count, " experts.");
    const size_t cuda_expert_count = cuda_eligible_expert_count - cpu_offload_expert_count_;

    std::sort(cuda_kernels.begin(), cuda_kernels.end(), [this](const KernelState* lhs, const KernelState* rhs) {
      if (cpu_offload_strategy_ != CpuOffloadStrategy::Adaptive && lhs->key.first == rhs->key.first &&
          lhs->topological_order != rhs->topological_order) {
        return lhs->topological_order < rhs->topological_order;
      }
      return lhs->key < rhs->key;
    });

    bool all_zero = true;
    for (const auto* state : cuda_kernels) {
      for (size_t expert = 0; expert < state->experts.count; ++expert) {
        all_zero = all_zero && counters_[state->experts.begin + expert] == 0.0;
      }
    }

    if (cpu_offload_strategy_ != CpuOffloadStrategy::Adaptive) {
      if (cpu_offload_layer_order_ == CpuOffloadLayerOrder::Last) {
        std::reverse(cuda_kernels.begin(), cuda_kernels.end());
      } else if (cpu_offload_layer_order_ == CpuOffloadLayerOrder::Middle && !cuda_kernels.empty()) {
        InlinedVector<KernelState*> ordered;
        ordered.reserve(cuda_kernels.size());
        const size_t middle = (cuda_kernels.size() - 1) / 2;
        for (size_t i = 0; i < cuda_kernels.size(); ++i) {
          const size_t index = i % 2 == 0 ? middle - i / 2 : middle + 1 + i / 2;
          ordered.push_back(cuda_kernels[index]);
        }
        cuda_kernels = std::move(ordered);
      }
      size_t target_capacity = 0;
      if (cpu_offload_strategy_ == CpuOffloadStrategy::Layered) {
        for (const auto* state : cuda_kernels) {
          target_capacity += static_cast<size_t>(cpu_offload_gpu_ratio_ * state->experts.count);
        }
      }
      // Exceed the per-layer target only when needed to keep exactly K experts on CPU.
      size_t excess = cuda_expert_count > target_capacity ? cuda_expert_count - target_capacity : 0;
      size_t remaining = cuda_expert_count;
      for (auto* state : cuda_kernels) {
        size_t capacity = state->experts.count;
        if (cpu_offload_strategy_ == CpuOffloadStrategy::Layered) {
          capacity = static_cast<size_t>(cpu_offload_gpu_ratio_ * state->experts.count);
          const size_t extra = std::min(excess, state->experts.count - capacity);
          capacity += extra;
          excess -= extra;
        }
        const size_t count = std::min(remaining, capacity);
        InlinedVector<int> experts;
        experts.reserve(state->experts.count);
        for (size_t expert = 0; expert < state->experts.count; ++expert) {
          experts.push_back(static_cast<int>(expert));
        }
        if (cpu_offload_strategy_ == CpuOffloadStrategy::Layered) {
          std::sort(experts.begin(), experts.end(), [&](int lhs, int rhs) {
            const double lhs_counter = counters_[state->experts.begin + static_cast<size_t>(lhs)];
            const double rhs_counter = counters_[state->experts.begin + static_cast<size_t>(rhs)];
            return lhs_counter != rhs_counter ? lhs_counter > rhs_counter : lhs < rhs;
          });
        }
        state->cuda_experts.reserve(count);
        state->cuda_experts.insert(state->cuda_experts.end(), experts.begin(), experts.begin() + count);
        std::sort(state->cuda_experts.begin(), state->cuda_experts.end());
        remaining -= count;
      }
      ORT_ENFORCE(remaining == 0);
    } else if (all_zero) {
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
    state.pilot.moe_expert_counter_alpha_ = alpha_;
    state.pilot.SetMoeCudaExperts(state.cuda_experts);
    state.pilot.moe_counters_ = gsl::make_span(counters_).subspan(state.experts.begin, state.experts.count);
  }

  initialized_ = true;
  return Status::OK();
}

KernelPilot* KernelPilotMoeExpertState::GetKernelPilot(const OpKernel* kernel) {
  const auto node = kernels_.find(kernel);
  return node != kernels_.end() ? &node->second.pilot : nullptr;
}

Status KernelPilotMoeExpertState::ScheduleSwaps() {
  struct SwapQueueMetrics {
    size_t in_flight{0};
    size_t blocked_candidates{0};
  };
  struct ScheduledSwap {
    const KernelState* state;
    int device_id;
    int outgoing_cuda_expert;
    int incoming_cpu_expert;
    double outgoing_score;
    double incoming_score;
  };
  std::map<int, SwapQueueMetrics> metrics_by_device;
  InlinedVector<ScheduledSwap> scheduled_swaps;
  InlinedVector<KernelState*> ordered_states;
  ordered_states.reserve(kernels_.size());
  for (auto& [kernel, state] : kernels_) {
    ORT_UNUSED_PARAMETER(kernel);
    auto* cache = state.pilot.GetMoeExpertCache();
    if (cache != nullptr) {
      ORT_RETURN_IF_ERROR(cache->ReclaimCompletedSwap());
      auto& metrics = metrics_by_device[cache->DeviceId()];
      if (cache->HasPendingSwap()) {
        ++metrics.in_flight;
      }
      ordered_states.push_back(&state);
    }
  }
  std::sort(ordered_states.begin(), ordered_states.end(),
            [](const KernelState* lhs, const KernelState* rhs) { return lhs->key < rhs->key; });

  for (auto* state : ordered_states) {
    auto* cache = state->pilot.GetMoeExpertCache();
    const int device_id = cache->DeviceId();
    auto& metrics = metrics_by_device[device_id];
    if (cache->HasPendingSwap() ||
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
      if (metrics.in_flight >= max_in_flight_swaps_per_device_) {
        ++metrics.blocked_candidates;
        continue;
      }
      ORT_RETURN_IF_ERROR(cache->StartSwap(coldest_cuda_expert, hottest_cpu_expert));
      scheduled_swaps.push_back(
          {state, device_id, coldest_cuda_expert, hottest_cpu_expert, cuda_min, cpu_max});
      ++metrics.in_flight;
    }
  }

  if (logging_logger_ != nullptr &&
      logging_logger_->OutputIsEnabled(logging::Severity::kINFO, logging::DataType::SYSTEM)) {
    size_t total_in_flight = 0;
    size_t total_blocked_candidates = 0;
    std::ostringstream event;
    event.imbue(std::locale::classic());
    event << "{\"request_id\":";
    common::WriteJsonString(event, logging_request_id_);
    event << ",\"devices\":[";
    bool first_device = true;
    for (const auto& [device_id, metrics] : metrics_by_device) {
      if (!first_device) {
        event << ",";
      }
      first_device = false;
      event << "{\"device_id\":" << device_id
            << ",\"in_flight\":" << metrics.in_flight
            << ",\"blocked_candidates\":" << metrics.blocked_candidates << "}";
      total_in_flight += metrics.in_flight;
      total_blocked_candidates += metrics.blocked_candidates;
    }
    event << "],\"scheduled_swaps\":[";
    for (size_t i = 0; i < scheduled_swaps.size(); ++i) {
      if (i != 0) {
        event << ",";
      }
      const auto& swap = scheduled_swaps[i];
      event << "{\"graph_scope\":";
      common::WriteJsonString(event, swap.state->key.first);
      event << ",\"node_index\":" << swap.state->key.second
            << ",\"device_id\":" << swap.device_id
            << ",\"outgoing_cuda_expert\":" << swap.outgoing_cuda_expert
            << ",\"incoming_cpu_expert\":" << swap.incoming_cpu_expert
            << ",\"outgoing_score\":" << std::setprecision(std::numeric_limits<double>::max_digits10)
            << swap.outgoing_score
            << ",\"incoming_score\":" << swap.incoming_score << "}";
    }
    event << "],\"total_in_flight\":" << total_in_flight
          << ",\"total_blocked_candidates\":" << total_blocked_candidates << "}";
    LOGS(*logging_logger_, INFO) << "moe_expert_swaps " << event.str();
  }
  return Status::OK();
}

Status KernelPilotMoeExpertState::RecordUsage(const OpKernel* kernel, gsl::span<const double> updated_counters) {
  ORT_RETURN_IF_NOT(initialized_, "MoE expert state is not initialized.");
  const auto node = kernels_.find(kernel);
  ORT_RETURN_IF(node == kernels_.end(), "Unknown MoE counter kernel.");
  const auto range = node->second.experts;
  ORT_RETURN_IF_NOT(updated_counters.empty() ||
                        (updated_counters.size() == range.count &&
                         std::all_of(updated_counters.begin(), updated_counters.end(),
                                     [](double value) { return std::isfinite(value) && value >= 0.0; })),
                    "Invalid MoE GPU expert counters.");
  const auto& usage = node->second.pilot.Moe();
  ORT_RETURN_IF_NOT(usage.ExpertCount() == range.count, "MoE kernel usage expert count does not match registration.");
  gsl::span<const int> used_expert_ids;
  ORT_RETURN_IF_ERROR(usage.GetSelectedExperts(used_expert_ids));
  gsl::span<const float> routing_probabilities;
  ORT_RETURN_IF_ERROR(usage.GetSelectedExpertProbabilities(routing_probabilities));
  ORT_RETURN_IF(counter_probability_weight_ > 0.0 &&
                    routing_probabilities.size() != used_expert_ids.size(),
                "MoE expert counter probability weighting requires routing probabilities.");
  const bool log_counter_event =
      logging_logger_ != nullptr &&
      logging_logger_->OutputIsEnabled(logging::Severity::kINFO, logging::DataType::SYSTEM);
  const auto demand_swaps = std::exchange(node->second.pilot.moe_demand_swaps_, std::nullopt);
  if (log_counter_event && demand_swaps) {
    std::ostringstream event;
    event << "{\"request_id\":";
    common::WriteJsonString(event, logging_request_id_);
    event << ",\"graph_scope\":";
    common::WriteJsonString(event, node->second.key.first);
    event << ",\"node_index\":" << node->second.key.second
          << ",\"missing_experts\":" << demand_swaps->missing
          << ",\"swapped_experts\":" << demand_swaps->swapped
          << ",\"all_required_on_cuda\":" << (demand_swaps->all_required_on_cuda ? "true" : "false") << "}";
    LOGS(*logging_logger_, INFO) << "moe_expert_demand_swaps " << event.str();
  }
  auto counters = gsl::make_span(counters_).subspan(range.begin, range.count);
  InlinedVector<double> logged_scores;
  InlinedVector<size_t> node_ranks;
  InlinedHashMap<int, size_t> selected_ranks;
  InlinedHashMap<int, size_t> global_positions;
  std::unique_lock<std::mutex> counter_logging_lock(counter_logging_mutex_, std::defer_lock);
  if (log_counter_event) {
    counter_logging_lock.lock();
    logged_scores.assign(counters.begin(), counters.end());
    InlinedVector<size_t> ranked_node_experts(range.count);
    std::iota(ranked_node_experts.begin(), ranked_node_experts.end(), 0);
    std::sort(ranked_node_experts.begin(), ranked_node_experts.end(),
              [counters](size_t lhs, size_t rhs) {
                return counters[lhs] != counters[rhs] ? counters[lhs] > counters[rhs] : lhs < rhs;
              });
    node_ranks.reserve(range.count);
    for (size_t expert = 0; expert < range.count; ++expert) {
      node_ranks.push_back(0);
    }
    for (size_t rank = 0; rank < ranked_node_experts.size(); ++rank) {
      node_ranks[ranked_node_experts[rank]] = rank;
    }

    InlinedVector<int> ranked_selected_experts(used_expert_ids.begin(), used_expert_ids.end());
    std::sort(ranked_selected_experts.begin(), ranked_selected_experts.end(),
              [counters](int lhs, int rhs) {
                const size_t lhs_index = static_cast<size_t>(lhs);
                const size_t rhs_index = static_cast<size_t>(rhs);
                return counters[lhs_index] != counters[rhs_index]
                           ? counters[lhs_index] > counters[rhs_index]
                           : lhs < rhs;
              });
    selected_ranks.reserve(ranked_selected_experts.size());
    for (size_t rank = 0; rank < ranked_selected_experts.size(); ++rank) {
      selected_ranks.emplace(ranked_selected_experts[rank], rank);
    }

    global_positions.reserve(used_expert_ids.size());
    for (int expert : used_expert_ids) {
      const size_t global_expert = expert_ids_.at({kernel, expert});
      const double score = counters_[global_expert];
      size_t global_position = 0;
      for (size_t other_expert = 0; other_expert < counters_.size(); ++other_expert) {
        if (counters_[other_expert] > score ||
            (counters_[other_expert] == score && other_expert < global_expert)) {
          ++global_position;
        }
      }
      global_positions.emplace(expert, global_position);
    }
  }
  if (!updated_counters.empty()) {
    std::copy(updated_counters.begin(), updated_counters.end(), counters.begin());
  } else {
    for (auto& counter : counters) {
      counter *= alpha_;
    }
    for (size_t i = 0; i < used_expert_ids.size(); ++i) {
      const double contribution =
          (1.0 - counter_probability_weight_) +
          counter_probability_weight_ * static_cast<double>(routing_probabilities.empty() ? 1.0f
                                                                                          : routing_probabilities[i]);
      counters_[expert_ids_.at({kernel, used_expert_ids[i]})] += (1.0 - alpha_) * contribution;
    }
  }

  if (log_counter_event) {
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
    event << ",\"moe_count\":" << kernels_.size()
          << ",\"total_expert_count\":" << counters_.size()
          << ",\"expert_offset\":" << range.begin;

    gsl::span<const int> cuda_experts;
    ORT_RETURN_IF_ERROR(node->second.pilot.GetMoeCudaExperts(cuda_experts));
    const bool has_expert_cache = node->second.pilot.GetMoeExpertCache() != nullptr;
    const bool node_runs_on_cuda = kernel->Node().GetExecutionProviderType() == kCudaExecutionProvider;

    event << ",\"selected_experts\":[";
    for (size_t i = 0; i < used_expert_ids.size(); ++i) {
      if (i != 0) {
        event << ",";
      }
      const int selected_expert = used_expert_ids[i];
      const size_t expert = static_cast<size_t>(selected_expert);
      const bool runs_on_cuda =
          has_expert_cache
              ? std::find(cuda_experts.begin(), cuda_experts.end(), selected_expert) != cuda_experts.end()
              : node_runs_on_cuda;
      event << "{\"expert_id\":" << selected_expert
            << ",\"score\":" << std::setprecision(std::numeric_limits<double>::max_digits10)
            << logged_scores[expert];
      if (!routing_probabilities.empty()) {
        event << ",\"routing_probability\":" << routing_probabilities[i];
      }
      event << ",\"selected_rank\":" << selected_ranks.at(selected_expert)
            << ",\"node_rank\":" << node_ranks[expert]
            << ",\"global_position\":" << global_positions.at(selected_expert)
            << ",\"device\":\"" << (runs_on_cuda ? "CUDA" : "CPU") << "\"}";
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

Status KernelPilotMoeExpertState::SavePopularity() const {
  std::ofstream output{std::filesystem::path(ToPathString(popularity_output_file_))};
  ORT_RETURN_IF_NOT(output.is_open(), "Unable to open MoE expert popularity file: ", popularity_output_file_);

  InlinedVector<const KernelState*> states;
  states.reserve(kernels_.size());
  for (const auto& [kernel, state] : kernels_) {
    ORT_UNUSED_PARAMETER(kernel);
    states.push_back(&state);
  }
  std::sort(states.begin(), states.end(),
            [](const KernelState* lhs, const KernelState* rhs) { return lhs->key < rhs->key; });

  output.imbue(std::locale::classic());
  output << "{\"alpha\":" << std::setprecision(std::numeric_limits<double>::max_digits10) << alpha_
         << ",\"epsilon\":" << swap_epsilon_ << ",\"experts\":[";
  bool first = true;
  for (size_t moe = 0; moe < states.size(); ++moe) {
    const auto& state = *states[moe];
    for (size_t expert = 0; expert < state.experts.count; ++expert) {
      if (!first) {
        output << ",";
      }
      first = false;
      output << "{\"moe\":" << moe
             << ",\"expert\":" << expert
             << ",\"popularity\":" << counters_[state.experts.begin + expert] << "}";
    }
  }
  output << "]}\n";
  ORT_RETURN_IF_NOT(output.good(), "Unable to write MoE expert popularity file: ", popularity_output_file_);
  return Status::OK();
}

}  // namespace onnxruntime
