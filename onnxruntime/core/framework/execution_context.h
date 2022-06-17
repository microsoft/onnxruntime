#pragma once
#include "core/framework/session_state.h"
#include "core/framework/execution_frame.h"
#include "core/framework/bfc_arena.h"

namespace onnxruntime {

struct ReleasePlan {
  std::unique_ptr<std::atomic_int[]> value_ref_counts_;
  std::unordered_map<OrtValueIndex, OrtValueIndex> reused_map_;
  std::unordered_map<onnxruntime::NodeIndex, std::vector<OrtValueIndex>> node_value_map_;
};

// execution context that support to execute a command on stream.
// The notifications got instantiated when execution context is constructed.
// TODO: if we merge the notifications to execution frame, we might don't need this.
struct ExecutionContext {
  const SessionState* session_state;
  std::unique_ptr<ExecutionFrame> frame;
  const logging::Logger* logger;
  std::vector<std::unique_ptr<synchronize::Notification>> notifications;
  std::unique_ptr<ReleasePlan> release_plan;
  std::vector<std::unique_ptr<Stream>> device_streams;

  ExecutionContext(const SessionState& sess_state,
                   const std::vector<std::unique_ptr<LogicStream>>& logic_streams,
                   std::vector<size_t> notification_owners,
                   const std::vector<int>& feed_mlvalue_idxs,
                   const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                   std::vector<OrtValue>& fetches,
                   const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                   const logging::Logger& sess_logger) : session_state(&sess_state),
                                                         logger(&sess_logger) {
    //1. bind logic stream to device stream;
    for (auto& logic_stream : logic_streams) {
      if (logic_stream->commands_.size() > 0) {
        auto& stream_handle_registry = sess_state.GetStreamHandleRegistryInstance();
        auto create_stream_fn = stream_handle_registry.GetCreateStreamFn(logic_stream->ep_->Type());
        device_streams.emplace_back(create_stream_fn(logic_stream->ep_));
      } else {
        device_streams.push_back(nullptr);
      }
    }

    for (auto i = 0; i < notification_owners.size(); ++i) {
      auto& stream = device_streams[notification_owners[i]];
      notifications.push_back(std::move(stream->CreateNotification(/*TODO: calculate num of consumers*/ 0)));
    }

    // create frame
    frame = std::make_unique<ExecutionFrame>(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, sess_state, &device_streams);
  }

  ~ExecutionContext() {
    for (auto& stream : device_streams) {
      if (stream) {
        auto& allocators = stream->provider->GetAllocators();
        for (auto& alloc : allocators) {
          if (alloc->Info().alloc_type == OrtArenaAllocator) {
            auto* arena_alloc = static_cast<BFCArena*>(alloc.get());
            auto* stream_aware_alloc = static_cast<StreamAwareArena*>(arena_alloc);
            if (stream_aware_alloc) {
              stream_aware_alloc->ReleaseStreamBuffers(stream.get());
            }
          }
        }
      }
    }
  }

  void RecycleNodeInputs(onnxruntime::NodeIndex node_index) {
    ORT_ENFORCE(frame);
    for (auto it : release_plan->node_value_map_[node_index]) {
      if (--release_plan->value_ref_counts_[it] == 0) {
        auto original_it = release_plan->reused_map_.find(it);
        if (original_it == release_plan->reused_map_.end()) {
          ORT_ENFORCE(frame->ReleaseMLValue(it).IsOK());
          LOGS(*logger, INFO) << "ort value " << it << " released";
        } else {
          if (--release_plan->value_ref_counts_[original_it->second] == 0) {
            ORT_ENFORCE(frame->ReleaseMLValue(original_it->second).IsOK());
            LOGS(*logger, INFO) << "ort value " << original_it->second << " released";
          }
        }
      }
    }
  }
};
}