// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/spin_pause.h"
#include "core/framework/parallel_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/framework/execution_frame.h"
#include "core/framework/utils.h"
#include "core/framework/mldata_type_utils.h"
#include "core/graph/constants.h"
#include "core/common/logging/macros.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/execution_context.h"

#ifdef USE_CUDA
#include <nvtx3/nvToolsExtCuda.h>
#endif

#include <vector>

namespace onnxruntime {

struct Barrier {
  std::atomic_bool set_{false};
  void set() {
    set_.store(true);
  }
  void wait() {
    while (!set_.load()) {
      onnxruntime::concurrency::SpinPause();
    }
  }
};

using NotificationIndex = size_t;

struct ParallelExecutionPlanImpl {
  ParallelExecutionPlanImpl(const SessionState& session_state);
  
  common::Status Execute(const SessionState& session_state,
                         const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds,
                         const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger);
  const SessionState& session_state_;
};

//todo: remove dependency on session_state

ParallelExecutionPlanImpl::ParallelExecutionPlanImpl(const SessionState& session_state) : session_state_(session_state) {}

common::Status ParallelExecutionPlanImpl::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                                  const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                                  std::vector<OrtValue>& fetches,
                                                  const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                                  const logging::Logger& logger) {
  auto* tp = session_state.GetInterOpThreadPool();
  auto* plan = session_state.GetExecutionPlan();
  // prepare the execution context, notifications got initialized.
  ExecutionContext execution_context(session_state, 
      plan->logic_streams, 
      plan->notification_owners_, 
      feed_mlvalue_idxs, 
      feeds, 
      fetch_mlvalue_idxs, 
      fetches, 
      fetch_allocators, 
      logger);
  // execution_context.release_plan = GenerateReleasePlan();
  std::unique_ptr<Barrier[]> barriers{new Barrier[plan->num_logic_streams_-1]}; // TODO: handle case when num_logic_streams_ == 0

  // WARNING: all task scheduled must be less or equal to the number 
  // of inter op threads, otherwise the execution might hang
  ORT_ENFORCE(plan->num_logic_streams_ <= concurrency::ThreadPool::DegreeOfParallelism(tp));

  for (int i = 0; i < plan->num_logic_streams_-1; ++i) {
    if (plan->logic_streams[i]->commands_.empty()) {
      barriers.get()[i].set();  // let go the stream if it is empty
    } else {
      concurrency::ThreadPool::Schedule(tp, [plan, i, this, &barriers, &execution_context]() {
        plan->logic_streams[i]->Run(execution_context);
        // flush
        execution_context.device_streams[i]->Flush();
        barriers.get()[i].set();
      });
    }
  }//for

  // run last stream in main thread
  LogicStream* stream = plan->logic_streams[plan->num_logic_streams_-1].get();
  stream->Run(execution_context);

  for (int i = 0; i < plan->num_logic_streams_-1; ++i) {
    barriers[i].wait();
  }

  //TODO: we might need to flush all the stream before return the result.
  ORT_RETURN_IF_ERROR(execution_context.frame->GetOutputs(fetches));
  return Status::OK();
}

ParallelExecutionPlan::ParallelExecutionPlan(const SessionState& session_state) {
  impl_ = std::make_unique<ParallelExecutionPlanImpl>(session_state);
}

common::Status ParallelExecutionPlan::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                              const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                              std::vector<OrtValue>& fetches,
                                              const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                              const logging::Logger& logger) {
  return impl_->Execute(session_state, feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, logger);
}

}  // namespace onnxruntime