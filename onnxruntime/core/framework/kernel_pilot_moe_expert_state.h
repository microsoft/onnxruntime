// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <istream>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <gsl/gsl>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/kernel_pilot.h"

namespace onnxruntime {

class OpKernel;
namespace logging {
class Logger;
}

// Session-global host policy and counters. CUDA allocations and transfer mechanics stay provider-side.
class KernelPilotMoeExpertState {
 public:
  // Counter-update logging
  // ----------------------

  KernelPilotMoeExpertState() = default;
  ~KernelPilotMoeExpertState();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelPilotMoeExpertState);

  static constexpr size_t kMaxCounterLogRecordsPerRun = 1024;

  // Starts one Run and optionally enables counter-update logging for it.
  // A second Run is rejected because selections and counters are session-owned state.
  Status BeginRun(std::string request_id, const logging::Logger* logger);

  // Releases the active Run and its optional logging state.
  Status EndRun(bool run_succeeded = true);

  // Expert counters
  // ---------------

  // Configures the exponential counter update:
  //   counter = alpha * counter + (1 - alpha) * selected
  // Must be called before registering any MoE kernels.
  Status SetCounterAlpha(double alpha);

  // Blends a selected expert's binary contribution with its routing probability.
  // Zero preserves binary counting; one uses only the routing probability.
  Status SetCounterProbabilityWeight(double probability_weight);
  double CounterProbabilityWeight() const noexcept { return counter_probability_weight_; }

  // Enables static CUDA placement and sets the global number of experts offloaded to CPU.
  // Must be called before FinalizeInitialization().
  Status SetCpuOffloadExpertCount(size_t cpu_offload_expert_count);

  // Basic and layered placement are QMoE-only; basic never swaps.
  Status SetCpuOffloadStrategy(std::string_view strategy);

  // Target GPU fraction per layer for layered placement, before preserving the exact global budget.
  Status SetCpuOffloadGpuRatio(double gpu_ratio);

  // Layer priority for GPU placement with basic/layered: first, last, or center-outward.
  Status SetCpuOffloadLayerOrder(std::string_view layer_order);

  // Sets the relative margin required before replacing a CUDA-resident expert.
  // Must be called before registering any MoE kernels.
  Status SetSwapEpsilon(double epsilon);

  // Sets the maximum concurrent adaptive swaps on each CUDA device.
  // Must be called before registering any MoE kernels. Zero disables adaptive swaps.
  Status SetMaxInFlightSwapsPerDevice(size_t max_in_flight_swaps_per_device);

  // Writes the final expert popularities as JSON when this session-owned state is destroyed.
  Status SetPopularityOutputFile(std::string output_file);

  // Registers one resolved MoE kernel and allocates its contiguous counter range and
  // provider-independent KernelPilot. Registration closes after initialization.
  Status RegisterNode(const OpKernel* kernel, std::string_view graph_scope, size_t node_index,
                      std::string_view node_type, size_t expert_count, size_t topological_order = 0);

  // Loads alpha, epsilon, and initial counters from the JSON produced by SavePopularity().
  // Nodes use their stable zero-based ordinal after sorting by graph scope and node index.
  // Omitted experts retain their current values. Invalid or duplicate records leave all
  // parameters and counters unchanged.
  Status Load(std::istream& input);

  // Closes registration and loading. Run-time counter updates require finalized state.
  Status FinalizeInitialization();

  // Returns the pilot owned by a registered kernel, or nullptr for any other kernel.
  KernelPilot* GetKernelPilot(const OpKernel* kernel);

  // During a Run, only this kernel may read its counters.
  Status GetCounters(const OpKernel* kernel, InlinedVector<double>& counters) const;

  // Maps a kernel-local expert ID to its index in the session-wide counter array.
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

  // Returns the total number of experts registered across all MoE kernels.
  size_t TotalExpertCount() const noexcept { return counters_.size(); }

 private:
  friend class KernelPilot;
  Status RecordUsage(const OpKernel* kernel, gsl::span<const double> updated_counters = {});
  Status ScheduleSwaps();
  Status SavePopularity() const;

  using Key = std::pair<std::string, size_t>;
  struct ExpertRange {
    size_t begin;
    size_t count;
  };
  struct KernelState {
    KernelState(Key key, ExpertRange range, KernelPilotMoeExpertState& owner, const OpKernel* kernel,
                size_t topological_order)
        : key(std::move(key)), experts(range), pilot(owner, kernel), topological_order(topological_order) {}
    Key key;
    ExpertRange experts;
    KernelPilot pilot;
    InlinedVector<int> cuda_experts;
    size_t topological_order;
  };
  NodeHashMap<const OpKernel*, KernelState> kernels_;
  InlinedHashMap<std::pair<const OpKernel*, int>, size_t> expert_ids_;
  InlinedVector<double> counters_;
  InlinedVector<KernelPilot*> deferred_usage_;
  std::mutex counter_logging_mutex_;
  mutable std::mutex run_mutex_;
  bool run_active_{false};
  std::string logging_request_id_;
  const logging::Logger* logging_logger_{nullptr};
  std::atomic<size_t> logging_record_count_{0};
  double alpha_{0.75};
  double counter_probability_weight_{0.0};
  std::string popularity_output_file_;
  size_t cpu_offload_expert_count_{0};
  double swap_epsilon_{0.01};
  size_t max_in_flight_swaps_per_device_{4};
  bool cpu_offload_enabled_{false};
  enum class CpuOffloadStrategy { Adaptive,
                                  Basic,
                                  Layered };
  CpuOffloadStrategy cpu_offload_strategy_{CpuOffloadStrategy::Adaptive};
  double cpu_offload_gpu_ratio_{0.8};
  enum class CpuOffloadLayerOrder { First,
                                    Last,
                                    Middle };
  CpuOffloadLayerOrder cpu_offload_layer_order_{CpuOffloadLayerOrder::First};
  bool initialized_{false};
};

}  // namespace onnxruntime
