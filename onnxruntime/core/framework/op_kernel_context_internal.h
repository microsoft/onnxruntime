// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
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
                                   Stream* stream)
      : OpKernelContext(&frame, &kernel, stream, session_state.GetThreadPool(), logger),
        session_state_(session_state),
        terminate_flag_(terminate_flag) {
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
  const bool& terminate_flag_;
  std::vector<const OrtValue*> implicit_input_values_;
};

}  // namespace onnxruntime
