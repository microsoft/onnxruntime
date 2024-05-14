// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/logging/logging.h"
#include "core/framework/device_stream_collection.h"
#include "core/framework/execution_frame.h"
#include "core/framework/ort_value.h"
#include "core/framework/iexecutor.h"
#include "core/framework/stream_handles.h"
#include "core/graph/basic_types.h"
#include "core/common/inlined_containers.h"
#include "core/framework/memory_info.h"
#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#endif

namespace onnxruntime {
class SessionState;

class SessionScope;
typedef InlinedHashMap<std::string, OrtValue> OrtValueCache;
typedef std::shared_ptr<OrtValueCache> OrtValueCachePtr;

// Execution context that support to execute a command on stream.
// It is composed by following components:
// 1. a execution frame
// 2. a collection of device stream instances that kernels can launch to.
// 3. a set of notification instances needed to perform synchronization in current execution plan.
class StreamExecutionContext {
 public:
  /*
   * LIMITATION:
   * CountDownBarrier is only for scenario that the v is set
   * to the # of consumers and each consumer calls Dec() exactly once.
   */
  class CountDownBarrier {
   public:
    CountDownBarrier() : v_{0} {};

    void Set(int32_t v) {
      ORT_ENFORCE(v >= 0);
      v_.store(v, std::memory_order_relaxed);
    }

    bool Dec() {
      return v_.fetch_sub(1, std::memory_order_relaxed) == 1;
    }

    int32_t Get() {
      return gsl::narrow_cast<int32_t>(v_.load(std::memory_order_relaxed));
    }

    void Inc() {
      ++v_;
    }

   private:
    std::atomic_int_fast32_t v_;
  };

  StreamExecutionContext(const SessionState& sess_state,
                         int32_t num_streams,
#ifdef ORT_ENABLE_STREAM
                         gsl::span<const size_t> notification_owners,
                         size_t num_barriers,
                         const DeviceStreamCollection* device_stream_map,
#endif
                         gsl::span<const int> feed_mlvalue_idxs,
                         gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                         const logging::Logger& sess_logger,
                         bool single_thread_mode);

  const SessionState& GetSessionState() const;

  const logging::Logger& GetLogger() const;

  ExecutionFrame& GetExecutionFrame();

  synchronize::Notification* GetNotification(size_t idx);

  void SetLogger(const logging::Logger& current_logger) {
    logger_ = &current_logger;
  }

  // Get status of the execution.
  // if one of the stream got non-OK status, the whole task status will be set as that non-OK status.
  const Status& TaskStatus() const;

  // Decrease the count of a given barrier.
  bool DecCountDownBarrier(size_t barrier_id);

  // The execution mode:
  // 1. single thread mode: all the streams will be launched using current host thread.
  // 2. multi-threads mode: use inter-op thread pool to schedule the N streams.
  bool SingleThreadMode() const { return single_thread_mode_; }

  // Get the Stream instance for a given logic sequence.
  // return nullptr if the device of given logic sequence doesn't register stream support.
  Stream* GetDeviceStream(size_t idx);

  // Decrease the count of remaining job by 1.
  void CompleteTask();

  // Increase the count of remaining job by 1.
  void AddTask();

  // This is only used under multi-threads mode.
  // blocked until all the jobs scheduled into inter-op thread pool complete.
  void WaitAll();

  // If one of the stream got non-OK status, update the status in the context.
  void SetStatus(Status& status);

  // Release the OrtValues after a step, based on the execution plan.
  void RecycleNodeInputs(onnxruntime::NodeIndex node_index);

#ifdef ENABLE_TRAINING
  void SetOrtValueCache(OrtValueCachePtr cache) {
    cache_ = std::move(cache);
  }

  OrtValueCachePtr& GetOrtValueCache() {
    return cache_;
  }
#endif

  ~StreamExecutionContext();

#ifdef ENABLE_TRAINING
  const ProgramRegion* GetCurrentRange() {
    return program_range_;
  }

  void SetCurrentRange(const ProgramRegion* range) {
    program_range_ = range;
  }

  const InlinedHashSet<NodeIndex>* GetNodeToExecute() {
    return node_to_execute_;
  }

  void SetNodeToExecute(const InlinedHashSet<NodeIndex>* node_to_execute) {
    node_to_execute_ = node_to_execute;
  }

#endif

 private:
  const SessionState* session_state_;

  ExecutionFrame frame_;

  const logging::Logger* logger_;

  std::unique_ptr<std::atomic_int[]> release_plan_;

  CountDownBarrier remain_tasks_;

  Status task_status_{Status::OK()};

#ifdef ENABLE_TRAINING
  const ProgramRegion* program_range_{nullptr};

  OrtValueCachePtr cache_{nullptr};

  // TODO: this is mainly for ort trainer
  // Should we deprecate it?
  const InlinedHashSet<NodeIndex>* node_to_execute_{nullptr};
#endif
  const bool single_thread_mode_;

#ifdef ORT_ENABLE_STREAM
  InlinedVector<std::unique_ptr<synchronize::Notification>> notifications_;
  // if it is nullptr, means current session doesn't have any EP using stream feature
  const DeviceStreamCollection* device_stream_map_;

  std::vector<CountDownBarrier> count_down_barriers_;
#endif
};

using NotificationIndex = size_t;

// Execute the stream at index 'stream_idx' with execution context 'ctx', from step 'since'.
void RunSince(size_t stream_idx,
              StreamExecutionContext& ctx,
              SessionScope& session_scope,
              const bool& terminate_flag,
              size_t since);

// Schedule the downstream jobs from other streams at 'trigger' step, based on the execution plan.
void ScheduleDownstream(StreamExecutionContext& ctx,
                        size_t trigger,
                        bool single_thread_mode,
                        const bool& terminate_flag,
                        SessionScope& session_scope);
}  // namespace onnxruntime
