// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#if !defined(ORT_MINIMAL_BUILD)
#include <algorithm>
#include "core/framework/execution_frame.h"
#include "core/framework/sequential_execution_plan.h"
#endif
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/session/onnxruntime_c_api.h"

// onnxruntime internal OpKernelContext derived class to provide additional
// APIs that aren't desirable to add to the public OpKernelContext API

namespace onnxruntime {
class SessionState;
class ExecutionFrame;

class OpKernelContextInternal : public OpKernelContext {
 public:
  explicit OpKernelContextInternal(const SessionState& session_state,
                                   IExecutionFrame& frame,
                                   const OpKernel& kernel,
                                   const logging::Logger& logger,
                                   const bool& terminate_flag,
                                   Stream* stream,
                                   profiling::Profiler* run_profiler = nullptr)
      : OpKernelContext(&frame, &kernel, stream, session_state.GetThreadPool(), logger),
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
    for (const auto* plan : active_workspace_plans_) {
      execution_frame_.ReleasePlannedWorkspace(plan->pattern_id, plan->location);
    }
  }

  Status GetPreallocatedWorkspace(int slot_id, size_t requested_bytes, void** workspace) override {
    *workspace = nullptr;
    const auto* execution_plan = session_state_.GetExecutionPlan();
    if (execution_plan == nullptr) {
      return Status::OK();
    }
    const auto node_plan = execution_plan->workspace_allocation_plan.find(GetNodeIndex());
    if (node_plan == execution_plan->workspace_allocation_plan.end()) {
      return Status::OK();
    }
    for (const auto& plan : node_plan->second) {
      if (plan.slot_id != slot_id || requested_bytes > plan.size_bytes) {
        continue;
      }
      ORT_RETURN_IF(std::find(active_workspace_plans_.begin(), active_workspace_plans_.end(), &plan) !=
                        active_workspace_plans_.end(),
                    "Workspace slot ", slot_id, " was requested more than once by node ", GetNodeIndex());
      ORT_RETURN_IF_ERROR(execution_frame_.GetPlannedWorkspace(
          plan.pattern_id, plan.location, plan.allocation_bytes, plan.alignment_bytes, workspace));
      active_workspace_plans_.push_back(&plan);
      return Status::OK();
    }
    LOGS(Logger(), VERBOSE) << "No declared workspace fits node " << GetNodeIndex()
                            << " slot=" << slot_id << " bytes=" << requested_bytes;
    return Status::OK();
  }
#endif

  bool GetUseDeterministicCompute() const override {
    return session_state_.GetUseDeterministicCompute();
  }

#if !defined(ORT_MINIMAL_BUILD)
  KernelPilot* GetKernelPilot() const override {
    kernel_pilot_ = session_state_.GetKernelPilot(GetKernel());
    return kernel_pilot_;
  }

  // Called by the executor only after successful Compute(). To commit usage, Compute() must obtain
  // the pilot via GetKernelPilot(), then BeginInvocation() and collect its selection before returning.
  // Without a lookup this is a no-op; KernelPilot::RecordUsage() also checks for a pending invocation.
  Status RecordKernelUsage() const {
    if (kernel_pilot_ == nullptr) {
      return Status::OK();
    }
    return kernel_pilot_->RecordUsage();
  }
#endif

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
  mutable KernelPilot* kernel_pilot_{nullptr};

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
  InlinedVector<const SequentialExecutionPlan::WorkspaceAllocationPlan*> active_workspace_plans_;
#endif
  const bool& terminate_flag_;
  profiling::Profiler* run_profiler_;
  std::vector<const OrtValue*> implicit_input_values_;
};

}  // namespace onnxruntime
