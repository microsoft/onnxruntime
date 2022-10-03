#include "core/framework/execution_context.h"
#include "core/framework/execution_provider.h"
#include "core/framework/execution_frame.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/session_state.h"
#include "core/common/spin_pause.h"

namespace onnxruntime {

class DeviceStreamCollectionImpl {
 public:
  DeviceStreamCollectionImpl(size_t num_streams, const SessionState& sess_state) : num_streams_(num_streams) {
    device_streams_.resize(num_streams, nullptr);
    auto& providers = sess_state.GetExecutionProviders();
    for (auto& ep : providers) {
      eps_.push_back(ep);
    }
    is_main_graph_ = sess_state.GetGraphViewer().ParentNode() == nullptr;
  }

  virtual ~DeviceStreamCollectionImpl() {
  }

  Status CleanUp() {
    for (auto& device_stream : device_streams_) {
      if (device_stream) {
        ORT_RETURN_IF_ERROR(device_stream->CleanUpOnRunEnd());
#ifndef ENABLE_TRAINING
        if (is_main_graph_) {
          device_stream->Flush();
        }
#endif
      }
    }
    // only clean the streams that is owned by current context
    for (auto& stream : device_streams_containers) {
      if (stream) {
        for (auto& ep : eps_) {
          auto& allocators = ep->GetAllocators();
          for (auto& alloc : allocators) {
            if (alloc->Info().device == stream->device &&
                alloc->Info().alloc_type == OrtArenaAllocator) {
              auto* arena_alloc = static_cast<BFCArena*>(alloc.get());
              auto* stream_aware_alloc = arena_alloc->AsStreamAwareAreana();
              if (stream_aware_alloc) {
                stream_aware_alloc->ReleaseStreamBuffers(stream.get());
              }
            }
          }
        }
      }
    }
    return Status::OK();
  }

  void SetDeviceStream(size_t idx, std::unique_ptr<Stream> stream) {
    ORT_ENFORCE(idx < num_streams_);
    device_streams_[idx] = stream.get();
    device_streams_containers.emplace_back(std::move(stream));
  }

  void SetDeviceStream(size_t idx, Stream* stream) {
    ORT_ENFORCE(idx < num_streams_);
    device_streams_[idx] = stream;
  }

  const std::vector<Stream*>& GetStreams() {
    return device_streams_;
  }

  size_t NumStreams() { return num_streams_; }

 private:
  size_t num_streams_;
  std::vector<Stream*> device_streams_;
  std::vector<std::unique_ptr<Stream>> device_streams_containers;
  // due to training's partial execution, the device streams collection may need to be hold
  // with a different lifetime of session state, we need to hold the reference of EPs.
  std::vector<std::shared_ptr<IExecutionProvider>> eps_;
  bool is_main_graph_ = false;
};

DeviceStreamCollection::DeviceStreamCollection(size_t num_streams, const SessionState& sess_state) : impl_(std::make_unique<DeviceStreamCollectionImpl>(num_streams, sess_state)) {}

DeviceStreamCollection::~DeviceStreamCollection() {}

void DeviceStreamCollection::SetDeviceStream(size_t idx, std::unique_ptr<Stream> stream) {
  impl_->SetDeviceStream(idx, std::move(stream));
}

void DeviceStreamCollection::SetDeviceStream(size_t idx, Stream* stream) {
  impl_->SetDeviceStream(idx, stream);
}

const std::vector<Stream*>& DeviceStreamCollection::GetStreams() const {
  return impl_->GetStreams();
}

size_t DeviceStreamCollection::NumStreams() const {
  return impl_->NumStreams();
}

Status DeviceStreamCollection::CleanUp() {
  return impl_->CleanUp();
}

ExecutionContext::ExecutionContext(const SessionState& sess_state,
                                   int32_t num_streams,
                                   gsl::span<const size_t> notification_owners,
                                   gsl::span<const int>& feed_mlvalue_idxs,
                                   gsl::span<const OrtValue>& feeds, gsl::span<const int>& fetch_mlvalue_idxs,
                                   gsl::span<const OrtValue> fetches,
                                   const InlinedHashMap<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                   size_t num_barriers,
                                   const logging::Logger& sess_logger,
                                   const DeviceStreamCollection& device_stream_map,
                                   bool single_thread_mode) : session_state(&sess_state),
                                                              logger(&sess_logger),
                                                              device_stream_map_(device_stream_map),
                                                              count_down_barriers_(num_barriers),
                                                              single_thread_mode_(single_thread_mode) {
  auto& device_streams = device_stream_map_.GetStreams();

  for (size_t i = 0; i < notification_owners.size(); ++i) {
    auto& stream = device_streams[notification_owners[i]];
    if (stream)
      notifications.emplace_back(stream->CreateNotification(/*TODO: calculate num of consumers*/ 0));
    else
      notifications.push_back(nullptr);
  }

  // create frame
  frame = std::make_unique<ExecutionFrame>(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, sess_state, &device_streams);

  // init barreris
  for (size_t i = 0; i < num_barriers; ++i) {
    count_down_barriers_[i].Set(2);
  }
  // init remain task to number of streams
  remain_tasks_.Set(num_streams);
  // generate release plan (the ref counts)
  auto& release_actions = session_state->GetExecutionPlan()->release_actions;
  release_plan = std::make_unique<ReleasePlan>();
  release_plan->value_ref_counts_.reset(new std::atomic_int[release_actions.size()]);
  for (size_t i = 0; i < release_actions.size(); ++i) {
    release_plan->value_ref_counts_[i] = static_cast<int>(release_actions[i].ref_count);
  }
}

const SessionState& ExecutionContext::GetSessionState() const { return *session_state; }

const logging::Logger& ExecutionContext::GetLogger() const { return *logger; }

ExecutionFrame* ExecutionContext::GetExecutionFrame() { return frame.get(); }

synchronize::Notification* ExecutionContext::GetNotification(size_t idx) { return notifications[idx].get(); }

const bool* ExecutionContext::TerminateFlag() const {
  if (!terminate_flag_)
    ORT_THROW("Terminate flag is not set");
  return terminate_flag_;
}

bool ExecutionContext::DecCountDownBarrier(size_t barrier_id) {
  return count_down_barriers_[barrier_id].Dec();
}

Stream* ExecutionContext::GetDeviceStream(size_t idx) {
  ORT_ENFORCE(idx < device_stream_map_.NumStreams());
  return device_stream_map_.GetStreams()[idx];
}

const Status& ExecutionContext::TaskStatus() const {
  return task_status_;
}

void ExecutionContext::CompleteTask() {
  remain_tasks_.Dec();
}

void ExecutionContext::AddTask() {
  remain_tasks_.Inc();
}

void ExecutionContext::WaitAll() {
  while (remain_tasks_.Get())
    onnxruntime::concurrency::SpinPause();
}

void ExecutionContext::SetStatus(Status& status) {
  // TODO: if multiple worker report non-ok status,
  // what is our strategy? currently we just keep
  // a random one. as long as it is not OK, the
  // execution will fail.
  if (task_status_.IsOK() && !status.IsOK())
    task_status_ = status;
}

ExecutionContext::~ExecutionContext() {}

void ExecutionContext::RecycleNodeInputs(onnxruntime::NodeIndex node_index) {
  ORT_ENFORCE(frame);
  auto* execution_plan = session_state->GetExecutionPlan();
  for (auto idx : execution_plan->node_release_list[node_index]) {
    if (--release_plan->value_ref_counts_[idx] == 0) {
      ORT_ENFORCE(frame->ReleaseMLValue(static_cast<int>(execution_plan->release_actions[idx].value_index)).IsOK());
      LOGS(*logger, INFO) << "ort value " << execution_plan->release_actions[idx].value_index << " released";
    }
  }
}

void RunSince(size_t stream_idx, ExecutionContext& ctx, size_t since) {
  if (!ctx.TaskStatus().IsOK()) {
    // already in bad status, terminate it
    ctx.CompleteTask();
    return;
  }

  // get logic stream
  auto& execution_plan = ctx.GetSessionState().GetExecutionPlan()->execution_plan;
  auto& logic_stream = execution_plan[stream_idx];
  size_t end = logic_stream->steps_.size();
#ifdef ENABLE_TRAINING
  auto* range = ctx.GetCurrentRange();
  if (range)
    end = std::min(end, range->stream_pc_range[stream_idx].second);
#endif

  if (since > end && since < logic_stream->steps_.size()) {
#ifdef ENABLE_TRAINING
    // this is a special handle for training
    // with ORTModule, we are partially execute the graph with a shared context.
    // there is a case that in forward pass we want to trigger downstream which
    // not in current range. We need to execute one step to consume the Barrier
    // counter otherwise later in backward the downstream won't execute correctly.
    // this is ugly, hopefully we won't need to worry about if deprecate ORTModule
    // by Torch Dynamo.
    if (!ctx.TaskStatus().IsOK()) {
      ctx.CompleteTask();
      return;
    }
    if (*ctx.TerminateFlag()) {
      Status status_made = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true.");
      ctx.SetStatus(status_made);
      ctx.CompleteTask();
      return;
    }
    bool continue_flag = true;
    Status status;
    ORT_TRY {
      status = logic_stream->steps_[since]->Execute(&ctx, stream_idx, continue_flag);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
    if (!status.IsOK()) {
      // terminate it
      ctx.SetStatus(status);
      ctx.CompleteTask();
      return;
    }
    if (continue_flag) {
      ORT_THROW("Execute the barrier step in backward range passed! this is not expected.");
    }
    ctx.CompleteTask();
    return;
#else
    ORT_THROW("Trigger execution beyond current range is not expected in inference build");
#endif
  }

  while (since < end) {
    if (!ctx.TaskStatus().IsOK()) {
      ctx.CompleteTask();
      return;
    }
    if (*ctx.TerminateFlag()) {
      Status status_made = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true.");
      ctx.SetStatus(status_made);
      ctx.CompleteTask();
      return;
    }
    bool continue_flag = true;
    Status status;
    ORT_TRY {
      status = logic_stream->steps_[since]->Execute(&ctx, stream_idx, continue_flag);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
    if (!status.IsOK()) {
      // terminate it
      ctx.SetStatus(status);
      ctx.CompleteTask();
      return;
    }
    if (!continue_flag) {
      // break but not terminate
      ctx.CompleteTask();
      return;
    }
    since++;
  }
  ORT_ENFORCE(since == end);
  ctx.CompleteTask();
  return;
}

void ScheduleDownstream(ExecutionContext& ctx,
                        size_t trigger,
                        bool single_thread_mode) {
  auto* ctx_ptr = &ctx;
  auto* plan = ctx.GetSessionState().GetExecutionPlan();
  auto& downstream_map = plan->downstream_map;
  auto* tp = single_thread_mode ? nullptr : ctx.GetSessionState().GetInterOpThreadPool();
  auto it = downstream_map.find(trigger);
  if (it != downstream_map.end()) {
    for (auto downstream : it->second) {
      // increase the task count before schedule down-stream
      ctx.AddTask();
      concurrency::ThreadPool::Schedule(tp,
                                        [ctx_ptr, downstream]() {
                                          RunSince(downstream.first, *ctx_ptr, downstream.second);
                                        });
    }
  }
}

}  // namespace onnxruntime
