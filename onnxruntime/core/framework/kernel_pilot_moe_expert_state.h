// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <istream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <gsl/gsl>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/kernel_pilot.h"
#include "core/framework/kernel_pilot_moe_logging_context.h"

namespace onnxruntime {

class OpKernel;
namespace logging {
class Logger;
}

// Host-side state only. CUDA allocations and placement policy do not belong here.
class KernelPilotMoeExpertState {
 public:
  // Routing logging
  // ---------------

  // Per-Run logging state. It keeps CUDA snapshots alive until execution-provider
  // synchronization completes and keeps concurrent Runs isolated.
  class LoggingContext final : public IKernelPilotMoeLoggingContext {
   public:
    static constexpr size_t kMaxMoeRoutingRecordsPerRun = 1024;
    static constexpr size_t kMaxMoeRoutingElementsPerRun = 2'000'000;

    LoggingContext(std::string request_id, const logging::Logger& logger);
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LoggingContext);

    // Metadata and timestamps used by CPU and CUDA routing records.
    const std::string& RequestId() const noexcept override;
    TimePoint StartProfiling() const override;
    uint64_t ProfilerStartTimeNs() const noexcept override;

    // Emits one structured moe_routing log entry. The routing arrays must already be
    // serialized as JSON; CPU records call this immediately and CUDA records call it
    // when the Run flushes its deferred snapshots.
    void RecordMoeRoutingEvent(const TimePoint& start_time,
                               const TimePoint& end_time,
                               std::string_view node_name,
                               NodeIndex node_index,
                               std::string_view node_type,
                               std::string expert_ids_json,
                               std::string router_weights_json,
                               int64_t num_rows,
                               int64_t top_k,
                               int execution_device_id,
                               int64_t completion_ns,
                               std::string_view completion_timestamp_source) const override;

    // Retains a provider-owned record until execution-provider synchronization has
    // completed, then FlushDeferredRecords() asks each record to emit its log entry.
    void AddDeferredRecord(std::unique_ptr<KernelPilotMoeDeferredRecord> record) const override;

    // Reserves space against the per-Run record and routing-element limits. Returns
    // false and records the dropped volume when either limit would be exceeded.
    bool TryReserveMoeRoutingRecord(size_t element_count) const override;

    // Logs one warning describing records rejected by TryReserveMoeRoutingRecord().
    void LogMoeStatisticsTruncation() const;

    // Emits and releases all deferred records, returning the first emission error.
    Status FlushDeferredRecords();

   private:
    std::string request_id_;
    const logging::Logger& logger_;
    uint64_t start_time_ns_;
    mutable std::mutex deferred_records_mutex_;
    mutable InlinedVector<std::unique_ptr<KernelPilotMoeDeferredRecord>> deferred_records_;
    mutable size_t moe_routing_record_count_{0};
    mutable size_t moe_routing_element_count_{0};
    mutable size_t dropped_moe_routing_record_count_{0};
    mutable size_t dropped_moe_routing_element_count_{0};
  };

  KernelPilotMoeExpertState() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelPilotMoeExpertState);

  // Starts logging for one Run. Concurrent Runs are rejected while logging is active
  // because the state owns a single current LoggingContext.
  Status BeginLogging(std::string request_id, const logging::Logger& logger);

  // Returns the active Run's provider-facing logging context, or nullptr when logging
  // is inactive.
  const IKernelPilotMoeLoggingContext* GetLoggingContext() const;

  // Flushes deferred records, logs truncation information, and releases the active
  // context so another Run can begin logging.
  Status EndLogging();

  // Expert counters
  // ---------------

  // Configures the exponential counter update:
  //   counter = alpha * counter + beta * selected
  // Must be called before registering any MoE kernels.
  Status SetCounterParameters(double alpha, double beta);

  // Registers one resolved MoE kernel and allocates its contiguous counter range and
  // provider-independent KernelPilot. Registration closes after initialization.
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

  // Closes registration and loading. Run-time counter updates require finalized state.
  Status FinalizeInitialization();

  // Returns the pilot owned by a registered kernel, or nullptr for any other kernel.
  KernelPilot* GetKernelPilot(const OpKernel* kernel);

  // Commit the kernel's collected usage after a successful invocation.
  Status RecordUsage(const OpKernel* kernel);

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
  using Key = std::pair<std::string, size_t>;
  struct ExpertRange {
    size_t begin;
    size_t count;
  };
  struct KernelState {
    KernelState(Key key, ExpertRange range) : key(std::move(key)), experts(range) {}
    Key key;
    ExpertRange experts;
    KernelPilot pilot;
  };
  NodeHashMap<const OpKernel*, KernelState> kernels_;
  InlinedHashMap<std::pair<const OpKernel*, int>, size_t> expert_ids_;
  InlinedVector<double> counters_;
  mutable std::mutex logging_context_mutex_;
  std::unique_ptr<LoggingContext> logging_context_;
  double alpha_{0.9};
  double beta_{0.1};
  bool initialized_{false};
};

}  // namespace onnxruntime
