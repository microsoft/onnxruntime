#include "core/framework/execution_context.h"
#include "core/framework/execution_provider.h"
#include "core/framework/execution_frame.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/session_state.h"
#include "core/common/spin_pause.h"

namespace onnxruntime {

class DeviceStreamCollectionImpl {
 public:
  DeviceStreamCollectionImpl(size_t num_streams) : num_streams_(num_streams) {
    device_streams_.resize(num_streams, nullptr);
  }

  virtual ~DeviceStreamCollectionImpl() {
    for (auto& device_stream : device_streams_) {
      if (device_stream) {
        device_stream->Flush();
      }
    }
    // only clean the streams that is owned by current context
    for (auto& stream : device_streams_containers) {
      if (stream) {
        auto& allocators = stream->provider->GetAllocators();
        for (auto& alloc : allocators) {
          if (alloc->Info().alloc_type == OrtArenaAllocator) {
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
};


DeviceStreamCollection::DeviceStreamCollection(size_t num_streams) : impl_(std::make_unique<DeviceStreamCollectionImpl>(num_streams)) {}

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

ExecutionContext::ExecutionContext(const SessionState& sess_state,
                                   int32_t num_streams,
                                   std::vector<size_t> notification_owners,
                                   gsl::span<const int>& feed_mlvalue_idxs,
                                   gsl::span<const OrtValue>& feeds, gsl::span<const int>& fetch_mlvalue_idxs,
                                   std::vector<OrtValue>& fetches,
                                   const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                   size_t num_barriers,
                                   const logging::Logger& sess_logger,
                                   const DeviceStreamCollection& device_streams_map,
                                   const bool& terminate_flag,
                                   bool single_thread_mode) : session_state(&sess_state),
                                                              logger(&sess_logger),
                                                              device_stream_map_(device_streams_map),
                                                              count_down_barriers_(num_barriers),
                                                              terminate_flag_(terminate_flag),
                                                              single_thread_mode_(single_thread_mode) {
  auto& device_streams = device_stream_map_.GetStreams();

  for (size_t i = 0; i < notification_owners.size(); ++i) {
    auto& stream = device_streams[notification_owners[i]];
    if (stream)
      notifications.emplace_back(std::move(stream->CreateNotification(/*TODO: calculate num of consumers*/ 0)));
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

const bool& ExecutionContext::TerminateFlag() const { return terminate_flag_; }

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
  if (range) {
    since = std::max(since, range->stream_pc_range[stream_idx].first);
    end = std::min(end, range->stream_pc_range[stream_idx].second);
  }
#endif

  while (since < end) {
    if (!ctx.TaskStatus().IsOK()) {
      ctx.CompleteTask();
      return;
    }
    if (ctx.TerminateFlag()) {
      Status status_made = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true.");
      ctx.SetStatus(status_made);
      ctx.CompleteTask();
      return;
    }
    bool continue_flag = true;
    Status status;
    ORT_TRY {
      status = logic_stream->steps_[since]->GetStepFun()(&ctx, stream_idx, continue_flag);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
    if (!status.IsOK()) {
      //terminate it
      ctx.SetStatus(status);
      ctx.CompleteTask();
      return;
    }
    if (!continue_flag) {
      //break but not terminate
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
    onnxruntime::NotificationIndex notification_index,
    bool single_thread_mode) {
  auto* ctx_ptr = &ctx;
  auto* plan = ctx.GetSessionState().GetExecutionPlan();
  auto& downstream_map = plan->downstream_map;
  auto* tp = single_thread_mode ? nullptr : ctx.GetSessionState().GetInterOpThreadPool();
  auto it = downstream_map.find(notification_index);
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

}
