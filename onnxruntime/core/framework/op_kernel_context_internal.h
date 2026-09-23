// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#if !defined(ORT_MINIMAL_BUILD)
#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "core/common/json_utils.h"
#include "core/framework/run_instrumentation.h"
#include "core/platform/env_var_utils.h"
#endif
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/session/onnxruntime_c_api.h"

// onnxruntime internal OpKernelContext derived class to provide additional
// APIs that aren't desirable to add to the public OpKernelContext API

namespace onnxruntime {
class SessionState;
class ExecutionFrame;

#if !defined(ORT_MINIMAL_BUILD)
// Holds per-run state for collecting MoE routing data without synchronizing after each kernel.
//
// A CUDA MoE kernel enqueues device-to-host copies of its routing outputs on the same stream
// that produced them, followed by a completion event. Stream ordering guarantees that the copies
// finish before the device scratch buffers can be reused by later work on that stream. The pinned
// host buffers and their CUDA events are owned by deferred records stored here, so they remain
// alive after the kernel returns.
//
// InferenceSession flushes the records after execution-provider OnRunEnd() performs the normal
// end-of-run synchronization. Each record also checks its completion event and synchronizes that
// event as a safety fallback before reading the host buffers and logging the routing decision.
//
// CUDA pinned allocations are arena-backed, but they cannot be returned to the arena while their
// copies are in flight. The routing record and element limits therefore bound the live pinned
// memory retained until the run is flushed.
class RunInstrumentationContext {
 public:
  static constexpr size_t kMaxMoeRoutingRecordsPerRun = 1024;
  static constexpr size_t kMaxMoeRoutingElementsPerRun = 2'000'000;

  RunInstrumentationContext(std::string request_id, const logging::Logger& logger)
      : request_id_(std::move(request_id)),
        logger_(logger),
        start_time_ns_(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count())) {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RunInstrumentationContext);

  const std::string& RequestId() const noexcept { return request_id_; }
  TimePoint StartProfiling() const { return std::chrono::high_resolution_clock::now(); }
  uint64_t ProfilerStartTimeNs() const noexcept { return start_time_ns_; }

  void RecordMoeRoutingEvent(const TimePoint&,
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

  void AddDeferredRecord(std::unique_ptr<DeferredRunInstrumentationRecord> record) const {
    std::lock_guard<std::mutex> lock(deferred_records_mutex_);
    deferred_records_.push_back(std::move(record));
  }

  bool TryReserveMoeRoutingRecord(size_t element_count) const {
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

  void LogMoeStatisticsTruncation() const {
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

  Status FlushDeferredRecords() {
    InlinedVector<std::unique_ptr<DeferredRunInstrumentationRecord>> records;
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

 private:
  std::string request_id_;
  const logging::Logger& logger_;
  uint64_t start_time_ns_;
  mutable std::mutex deferred_records_mutex_;
  mutable InlinedVector<std::unique_ptr<DeferredRunInstrumentationRecord>> deferred_records_;
  mutable size_t moe_routing_record_count_{0};
  mutable size_t moe_routing_element_count_{0};
  mutable size_t dropped_moe_routing_record_count_{0};
  mutable size_t dropped_moe_routing_element_count_{0};
};
#endif  // !defined(ORT_MINIMAL_BUILD)

class OpKernelContextInternal : public OpKernelContext {
 public:
  explicit OpKernelContextInternal(const SessionState& session_state,
                                   IExecutionFrame& frame,
                                   const OpKernel& kernel,
                                   const logging::Logger& logger,
                                   const bool& terminate_flag,
                                   Stream* stream,
                                   profiling::Profiler* run_profiler = nullptr
#if !defined(ORT_MINIMAL_BUILD)
                                   ,
                                   const RunInstrumentationContext* run_instrumentation_context = nullptr)
#else
                                   )
#endif
      : OpKernelContext(&frame, &kernel, stream, session_state.GetThreadPool(), logger
#if !defined(ORT_MINIMAL_BUILD)
                        ,
                        run_instrumentation_context),
#else
                        ),
#endif
        session_state_(session_state),
#if !defined(ORT_MINIMAL_BUILD)
        execution_frame_(frame),
#endif
        terminate_flag_(terminate_flag),
        run_profiler_(run_profiler) {
    const auto& implicit_inputs = kernel.Node().ImplicitInputDefs();
    int num_implicit_inputs = static_cast<int>(implicit_inputs.size());
    implicit_input_values_.reserve(num_implicit_inputs);

    for (int i = 0; i < num_implicit_inputs; ++i) {
      const auto* entry = GetImplicitInputMLValue(i);
      ORT_ENFORCE(entry != nullptr, "All implicit inputs should have OrtValue instances by now. ",
                  implicit_inputs[i]->Name(), " does not.");
      implicit_input_values_.push_back(entry);
    }

#if !defined(ORT_MINIMAL_BUILD)
    if (session_state_.GetNodeStatsRecorder() != nullptr) {
      auto alloc = OpKernelContext::GetAllocator(kernel.GetDevice(OrtMemTypeDefault));
      if (alloc != nullptr) {
        accounting_allocator_ = std::make_shared<AccountingAllocator>(std::move(alloc));
      }
    }
#endif
  }

#if !defined(ORT_MINIMAL_BUILD)
  ~OpKernelContextInternal() override {
    for (const auto* workspace_plan : active_workspace_plans_) {
      execution_frame_.ReleasePlannedWorkspace(workspace_plan->pattern_id, workspace_plan->location);
    }
  }

  Status GetPreallocatedWorkspaceRegion(int slot_id, size_t requested_bytes,
                                        WorkspaceBufferRegion& workspace) override {
    workspace = {};
    static const bool trace_workspace_lookup =
        ParseEnvironmentVariableWithDefault<int>(
            "ORT_MATMULNBITS_TRACE_LEGACY_WORKSPACE", 0) != 0;
    static std::atomic<bool> logged_no_execution_plan{false};
    static std::atomic<bool> logged_no_node_plan{false};
    static std::atomic<bool> logged_no_matching_slot{false};
    static std::atomic<bool> logged_frame_null{false};
    static std::atomic<bool> logged_success{false};

    const auto* execution_plan = session_state_.GetExecutionPlan();
    if (execution_plan == nullptr) {
      if (trace_workspace_lookup &&
          !logged_no_execution_plan.exchange(true, std::memory_order_relaxed)) {
        std::cerr << "[workspace_plan_lookup] state=no_execution_plan"
                  << " node=" << GetNodeIndex()
                  << " slot=" << slot_id
                  << " requested_bytes=" << requested_bytes
                  << std::endl;
      }
      return Status::OK();
    }

    const auto node_it = execution_plan->workspace_allocation_plan.find(GetNodeIndex());
    if (node_it == execution_plan->workspace_allocation_plan.end()) {
      if (trace_workspace_lookup &&
          !logged_no_node_plan.exchange(true, std::memory_order_relaxed)) {
        std::cerr << "[workspace_plan_lookup] state=no_node_plan"
                  << " node=" << GetNodeIndex()
                  << " slot=" << slot_id
                  << " requested_bytes=" << requested_bytes
                  << std::endl;
      }
      return Status::OK();
    }

    for (const auto& workspace_plan : node_it->second) {
      if (workspace_plan.slot_id != slot_id || requested_bytes > workspace_plan.size_bytes) {
        continue;
      }

      ORT_RETURN_IF(std::find(active_workspace_plans_.begin(), active_workspace_plans_.end(), &workspace_plan) !=
                        active_workspace_plans_.end(),
                    "Workspace slot ", slot_id, " was requested more than once by node ", GetNodeIndex());
      ORT_RETURN_IF_ERROR(execution_frame_.GetPlannedWorkspace(
          workspace_plan.pattern_id, workspace_plan.location,
          workspace_plan.allocation_bytes, workspace_plan.alignment_bytes, workspace));
      if (trace_workspace_lookup) {
        auto& logged = workspace.buffer == nullptr ? logged_frame_null : logged_success;
        if (!logged.exchange(true, std::memory_order_relaxed)) {
          std::cerr << "[workspace_plan_lookup] state="
                    << (workspace.buffer == nullptr ? "frame_returned_null" : "success")
                    << " node=" << GetNodeIndex()
                    << " slot=" << slot_id
                    << " requested_bytes=" << requested_bytes
                    << " planned_bytes=" << workspace_plan.size_bytes
                    << " allocation_bytes=" << workspace_plan.allocation_bytes
                    << " pattern_id=" << workspace_plan.pattern_id
                    << std::endl;
        }
      }
      active_workspace_plans_.push_back(&workspace_plan);
      if (workspace.buffer != nullptr && workspace.size_bytes < requested_bytes) {
        LOGS(Logger(), WARNING)
            << "Planned workspace slot " << slot_id << " for node " << GetNodeIndex()
            << " has " << workspace.size_bytes << " usable bytes but " << requested_bytes
            << " bytes were requested. Falling back to dynamic allocation.";
        workspace = {};
      }
      return Status::OK();
    }

    if (trace_workspace_lookup &&
        !logged_no_matching_slot.exchange(true, std::memory_order_relaxed)) {
      std::cerr << "[workspace_plan_lookup] state=no_matching_slot"
                << " node=" << GetNodeIndex()
                << " slot=" << slot_id
                << " requested_bytes=" << requested_bytes
                << " declared_slots=" << node_it->second.size()
                << std::endl;
    }
    return Status::OK();
  }
#endif

  bool GetUseDeterministicCompute() const override {
    return session_state_.GetUseDeterministicCompute();
  }

  const SessionState* SubgraphSessionState(const std::string& attribute_name) {
    return session_state_.GetSubgraphSessionState(GetNodeIndex(), attribute_name);
  }

  const OrtValue* GetInputMLValue(int index) const override {
    return OpKernelContext::GetInputMLValue(index);
  }

  OrtValue* GetOutputMLValue(int index) {
    return OpKernelContext::GetOutputMLValue(index);
  }

  OrtValue* GetPreallocatedOutputMLValue(int index) const {
    return OpKernelContext::GetPreallocatedOutputMLValue(index);
  }

#ifdef ENABLE_ATEN
  Status SetOutputMLValue(int index, const OrtValue& ort_value) {
    return OpKernelContext::SetOutputMLValue(index, ort_value);
  }
#endif

  OrtValue* OutputMLValue(int index, const TensorShape& shape) override {
    return OpKernelContext::OutputMLValue(index, shape);
  }

  // Get the OrtValue's for all implicit inputs. Order is same as Node::ImplicitInputDefs(). No nullptr entries.
  const std::vector<const OrtValue*>& GetImplicitInputs() const {
    return implicit_input_values_;
  }

  int GetOrtValueIndexForOutput(int output_index) const override {
    return OpKernelContext::GetOrtValueIndexForOutput(output_index);
  }

#if !defined(ORT_MINIMAL_BUILD)
  Status GetTempSpaceAllocator(AllocatorPtr* output) const override {
    if (accounting_allocator_) {
      *output = accounting_allocator_;
      return Status::OK();
    }
    return OpKernelContext::GetTempSpaceAllocator(output);
  }
#endif

#if !defined(ORT_MINIMAL_BUILD)
  bool GetAllocatorStats(AllocatorStats& stats) {
    if (accounting_allocator_ == nullptr) {
      return false;
    }
    accounting_allocator_->GetStats(&stats);
    return true;
  }
#endif

  const bool& GetTerminateFlag() const noexcept { return terminate_flag_; }

  profiling::Profiler* GetRunProfiler() const noexcept { return run_profiler_; }

 private:
#if !defined(ORT_MINIMAL_BUILD)
  class AccountingAllocator : public IAllocator {
   public:
    AccountingAllocator(AllocatorPtr alloc) : IAllocator(alloc->Info()), allocator_(std::move(alloc)) {
    }

    void* Alloc(size_t size) override {
      void* p = allocator_->Alloc(size);
      if (p != nullptr) {
        stats_.total_allocated_bytes += size;
      }
      return p;
    }

    void Free(void* p) override {
      allocator_->Free(p);
    }

    void GetStats(AllocatorStats* stats) override {
      *stats = stats_;
    }

   private:
    AllocatorPtr allocator_;
    AllocatorStats stats_;
  };

  AllocatorPtr accounting_allocator_;
#endif

  const SessionState& session_state_;
#if !defined(ORT_MINIMAL_BUILD)
  IExecutionFrame& execution_frame_;
#endif
  const bool& terminate_flag_;
  profiling::Profiler* run_profiler_;
  std::vector<const OrtValue*> implicit_input_values_;
#if !defined(ORT_MINIMAL_BUILD)
  InlinedVector<const SequentialExecutionPlan::WorkspaceAllocationPlan*> active_workspace_plans_;
#endif
};

}  // namespace onnxruntime
