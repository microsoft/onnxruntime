// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/stream_execution_context.h"
#include "core/framework/execution_provider.h"
#include "core/framework/execution_frame.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/session_state.h"

namespace onnxruntime {
void StreamExecutionContext::FirstFailureStatus::SetStatus(const Status& status) {
  if (status.IsOK()) {
    return;
  }

  std::lock_guard lock(mutex_);
  if (status_.IsOK()) {
    status_ = status;
  }
}

Status StreamExecutionContext::FirstFailureStatus::GetStatus() const {
  std::lock_guard lock(mutex_);
  return status_;
}

void StreamExecutionContext::FirstFailureStatus::Reset() {
  std::lock_guard lock(mutex_);
  status_ = Status::OK();
}

#ifdef ORT_ENABLE_STREAM
StreamExecutionContext::StreamExecutionContext(const SessionState& sess_state,
                                               int32_t num_streams,
                                               gsl::span<const size_t> notification_owners,
                                               size_t num_barriers,
                                               const DeviceStreamCollection* device_stream_map,
                                               gsl::span<const int> feed_mlvalue_idxs,
                                               gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                               std::vector<OrtValue>& fetches,
                                               const std::unordered_map<size_t, IExecutor::CustomAllocator>&
                                                   fetch_allocators,
                                               const logging::Logger& sess_logger,
                                               onnxruntime::CancellationToken terminate_token,
                                               bool single_thread_mode)
    : session_state_(&sess_state),
      frame_(feed_mlvalue_idxs,
             feeds,
             fetch_mlvalue_idxs,
             fetches,
             fetch_allocators,
             device_stream_map,
             sess_state),
      logger_(&sess_logger),
      single_thread_mode_(single_thread_mode),
      device_stream_map_(device_stream_map),
      count_down_barriers_(num_barriers) {
  notifications_.reserve(notification_owners.size());
  for (size_t i = 0; i < notification_owners.size(); ++i) {
    auto* stream = device_stream_map_ ? device_stream_map_->GetStream(notification_owners[i]) : nullptr;
    if (stream)
      notifications_.emplace_back(stream->CreateNotification(/*TODO: calculate num of consumers*/ 0));
    else
      notifications_.push_back(nullptr);
  }
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 26409 26400)
#endif
  std::atomic_int* p_release_plan_buffer = new std::atomic_int[sess_state.GetExecutionPlan()->release_actions.size()];
  release_plan_ = std::unique_ptr<std::atomic_int[]>(p_release_plan_buffer);
#ifdef _WIN32
#pragma warning(pop)
#endif

  // init barriers
  // one for the producer node: BarrierStep in execution_plan[i]->steps_
  // one for the downstream node: run via plan_.downstream_map
  for (size_t i = 0; i < num_barriers; ++i) {
    count_down_barriers_[i].Set(2);
  }
  ResetForExecution(num_streams, terminate_token);
  // generate release plan (the ref counts)
  auto& release_actions = sess_state.GetExecutionPlan()->release_actions;
  for (size_t i = 0; i < release_actions.size(); ++i) {
    release_plan_[i] = static_cast<int>(release_actions[i].ref_count);
  }
}

synchronize::Notification* StreamExecutionContext::GetNotification(size_t idx) {
  return notifications_[idx].get();
}

bool StreamExecutionContext::DecCountDownBarrier(size_t barrier_id) {
  return count_down_barriers_[barrier_id].Dec();
}

Stream* StreamExecutionContext ::GetDeviceStream(size_t idx) {
  if (device_stream_map_) {
    ORT_ENFORCE(idx < device_stream_map_->NumStreams());
    return device_stream_map_->GetStream(idx);
  } else {
    return nullptr;
  }
}

#else
StreamExecutionContext::StreamExecutionContext(const SessionState& sess_state,
                                               int32_t num_streams,
                                               gsl::span<const int> feed_mlvalue_idxs,
                                               gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                               std::vector<OrtValue>& fetches,
                                               const std::unordered_map<size_t, IExecutor::CustomAllocator>&
                                                   fetch_allocators,
                                               const logging::Logger& sess_logger,
                                               onnxruntime::CancellationToken terminate_token,
                                               bool single_thread_mode)
    : session_state_(&sess_state),
      frame_(feed_mlvalue_idxs,
             feeds,
             fetch_mlvalue_idxs,
             fetches,
             fetch_allocators,
             sess_state),
      logger_(&sess_logger),
      single_thread_mode_(single_thread_mode) {
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 26409 26400)
#endif
  std::atomic_int* p_release_plan_buffer = new std::atomic_int[sess_state.GetExecutionPlan()->release_actions.size()];
  release_plan_ = std::unique_ptr<std::atomic_int[]>(p_release_plan_buffer);
#ifdef _WIN32
#pragma warning(pop)
#endif
  ResetForExecution(num_streams, terminate_token);
  // generate release plan (the ref counts)
  auto& release_actions = sess_state.GetExecutionPlan()->release_actions;
  for (size_t i = 0; i < release_actions.size(); ++i) {
    release_plan_[i] = static_cast<int>(release_actions[i].ref_count);
  }
}

synchronize::Notification* StreamExecutionContext ::GetNotification(size_t /*idx*/) {
  ORT_THROW("Try to get notification in a build which doesn't enable Stream!");
}

bool StreamExecutionContext ::DecCountDownBarrier(size_t /*barrier_id*/) {
  ORT_THROW("Try to decrease barrier in a build which doesn't enable Stream!");
}

Stream* StreamExecutionContext ::GetDeviceStream(size_t /*idx*/) {
  return nullptr;
}
#endif

const SessionState& StreamExecutionContext ::GetSessionState() const { return *session_state_; }

const logging::Logger& StreamExecutionContext ::GetLogger() const { return *logger_; }

ExecutionFrame& StreamExecutionContext ::GetExecutionFrame() { return frame_; }

Status StreamExecutionContext ::TaskStatus() const {
  return task_status_.GetStatus();
}

void StreamExecutionContext ::CompleteTask() {
  remain_tasks_.Dec();
}

void StreamExecutionContext ::AddTask() {
  remain_tasks_.Inc();
}

void StreamExecutionContext ::WaitAll() {
  remain_tasks_.Wait();
}

void StreamExecutionContext ::SetStatus(const Status& status) {
  if (!status.IsOK()) {
    task_status_.SetStatus(status);
    stop_source_.request_stop();
  }
}

void StreamExecutionContext::ResetForExecution(int32_t num_tasks, onnxruntime::CancellationToken terminate_token) {
  ORT_ENFORCE(remain_tasks_.Get() == 0);

  external_stop_callback_.reset();
  stop_source_ = onnxruntime::CancellationSource{};
  task_status_.Reset();
  remain_tasks_.Set(num_tasks);
  external_stop_callback_.emplace(terminate_token, RequestStop{stop_source_});
}

StreamExecutionContext::~StreamExecutionContext() {}

void StreamExecutionContext::RecycleNodeInputs(onnxruntime::NodeIndex node_index) {
  auto* execution_plan = session_state_->GetExecutionPlan();
  for (auto idx : execution_plan->node_release_list[node_index]) {
    if (--release_plan_[idx] == 0) {
      ORT_ENFORCE(frame_.ReleaseMLValue(static_cast<int>(execution_plan->release_actions[idx].value_index)).IsOK());
      VLOGS(*logger_, 0) << "ort value " << execution_plan->release_actions[idx].value_index << " released";
    }
  }
}

void RunSince(size_t stream_idx, StreamExecutionContext& ctx, SessionScope& session_scope,
              onnxruntime::CancellationToken terminate_token, size_t since) {
  [[maybe_unused]] auto complete_task = gsl::finally([&ctx]() { ctx.CompleteTask(); });

  if (terminate_token.stop_requested()) {
    ctx.SetStatus(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true."));
    return;
  }

#ifdef USE_CANN
  // Leave it to CANN EP to fill the gap if they want to use run_options
  static onnxruntime::RunOptions run_options;
  // For CANN EP, it is necessary to explicitly create a corresponding Context for each thread in the thread pool,
  // which is different from CUDA Runtime API, but similar to CUDA Driver API.
  auto& execution_providers = ctx.GetSessionState().GetExecutionProviders();
  for (auto& xp : execution_providers) {
    auto status = xp->OnRunStart(run_options);
    if (!status.IsOK()) {
      ctx.SetStatus(status);
      return;
    }
  }
#endif

  // get logic stream
  auto& execution_plan = ctx.GetSessionState().GetExecutionPlan()->execution_plan;
  auto& logic_stream = execution_plan[stream_idx];
  size_t end = logic_stream->steps_.size();
#ifdef ENABLE_TRAINING
  auto* range = ctx.GetCurrentRange();
  if (range)
    end = std::min(end, range->stream_pc_range[stream_idx].second);
#endif

#ifdef ORT_ENABLE_STREAM
  // If the device stream has corresponding SetDevice function registered, it means GPU device should be properly set to the correct device.
  // The reason SetDevice should be called here is:
  //  - RunSince function can be invoked from a new thread
  //  - new threads default to using device 0, but the session may be tightly bound to a device > 0.
  auto device_stream = ctx.GetDeviceStream(stream_idx);
  if (device_stream) {
    auto set_device_fn = ctx.GetSessionState().GetStreamHandleRegistryInstance().GetSetDeviceFn(device_stream->GetDevice().Type());
    if (set_device_fn.has_value()) {
      auto device_id = device_stream->GetDevice().Id();
      set_device_fn.value()(device_id);
    }
  }
#endif

  while (since < end) {
    if (terminate_token.stop_requested()) {
      ctx.SetStatus(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true."));
      return;
    }
    bool continue_flag = true;
    Status status;
    ORT_TRY {
      status = logic_stream->steps_[since]->Execute(ctx, stream_idx, session_scope, terminate_token, continue_flag);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
    if (!status.IsOK()) {
      ctx.SetStatus(status);
      return;
    }
    if (!continue_flag) {
      return;
    }
    since++;
  }
  ORT_ENFORCE(since == end);
}

void ScheduleDownstream(StreamExecutionContext& ctx, size_t trigger, bool single_thread_mode,
                        onnxruntime::CancellationToken terminate_token, SessionScope& session_scope) {
  auto* plan = ctx.GetSessionState().GetExecutionPlan();
  auto& downstream_map = plan->downstream_map;
  auto* tp = single_thread_mode ? nullptr : ctx.GetSessionState().GetInterOpThreadPool();
  auto it = downstream_map.find(trigger);
  if (it != downstream_map.end()) {
    for (auto downstream : it->second) {
      // increase the task count before schedule down-stream
      ctx.AddTask();
      concurrency::ThreadPool::Schedule(tp, [&ctx, downstream, terminate_token, &session_scope]() {
        RunSince(downstream.first, ctx, session_scope, terminate_token, downstream.second);
      });
    }
  }
}

}  // namespace onnxruntime
