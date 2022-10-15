#pragma once
#include "core/common/logging/logging.h"
#include "core/framework/ort_value.h"
#include "core/framework/iexecutor.h"
#include "core/framework/stream_handles.h"
#include "core/graph/basic_types.h"
#include "core/common/inlined_containers.h"
#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#endif

namespace onnxruntime {
class SessionState;
class IExecutionFrame;
class ExecutionFrame;

using OrtValueIndex = int;

struct ReleasePlan {
  std::unique_ptr<std::atomic_int[]> value_ref_counts_;
};

class DeviceStreamCollectionImpl;
class DeviceStreamCollection {
 public:
  DeviceStreamCollection(size_t num_streams, const SessionState& sess_state);
  ~DeviceStreamCollection();
  void SetDeviceStream(size_t, std::unique_ptr<Stream> stream);
  void SetDeviceStream(size_t, Stream* stream);
  const std::vector<Stream*>& GetStreams() const;
  size_t NumStreams() const;
  Status CleanUp();

 private:
  std::unique_ptr<DeviceStreamCollectionImpl> impl_;
};

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

  int32_t Get() { return v_.load(std::memory_order_relaxed); }

  void Inc() {
    ++v_;
  }

 private:
  std::atomic_int_fast32_t v_;
};

class SessionScope;
typedef InlinedHashMap<std::string, OrtValue> OrtValueCache;
typedef std::shared_ptr<OrtValueCache> OrtValueCachePtr;

// execution context that support to execute a command on stream.
// The notifications got instantiated when execution context is constructed.
// TODO: if we merge the notifications to execution frame, we might don't need this.
class ExecutionContext {
 public:
  ExecutionContext(const SessionState& sess_state,
                   int32_t num_streams,
                   gsl::span<const size_t> notification_owners,
                   gsl::span<const int> feed_mlvalue_idxs,
                   gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                   std::vector<OrtValue>& fetches,
                   const InlinedHashMap<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                   size_t num_barriers,
                   const logging::Logger& sess_logger,
                   const DeviceStreamCollection& device_stream_map,
                   bool single_thread_mode);

  const SessionState& GetSessionState() const;

  const logging::Logger& GetLogger() const;

  ExecutionFrame* GetExecutionFrame();

  synchronize::Notification* GetNotification(size_t idx);

  const bool* TerminateFlag() const;

  void SetLogger(const logging::Logger& current_logger) {
    logger = &current_logger;
  }

  void SetTerminateFlag(const bool* terminate_flag) {
    terminate_flag_ = terminate_flag;
  }

  const Status& TaskStatus() const;

  bool DecCountDownBarrier(size_t barrier_id);

  bool SingleThreadMode() const { return single_thread_mode_; }

  Stream* GetDeviceStream(size_t idx);

  void CompleteTask();

  void AddTask();

  void WaitAll();

  void SetStatus(Status& status);

  void RecycleNodeInputs(onnxruntime::NodeIndex node_index);

#ifdef ENABLE_TRAINING
  void SetOrtValueCache(OrtValueCachePtr cache) {
    cache_ = std::move(cache);
  }

  OrtValueCachePtr& GetOrtValueCache() {
    return cache_;
  }
#endif

  ~ExecutionContext();

  SessionScope* GetSessionScope() {
    return session_scope_;
  }

  void SetSessionScope(SessionScope* session_scope) {
    session_scope_ = session_scope;
  }

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
  const SessionState* session_state;
  std::unique_ptr<ExecutionFrame> frame;
  const logging::Logger* logger;
  InlinedVector<std::unique_ptr<synchronize::Notification>> notifications;
  std::unique_ptr<ReleasePlan> release_plan;
  const DeviceStreamCollection& device_stream_map_;
  std::vector<CountDownBarrier> count_down_barriers_;
  CountDownBarrier remain_tasks_;
  const bool* terminate_flag_ = nullptr;
  Status task_status_{Status::OK()};
  SessionScope* session_scope_{};
#ifdef ENABLE_TRAINING
  const ProgramRegion* program_range_{nullptr};
  OrtValueCachePtr cache_{nullptr};
  // TODO: this is mainly for ort trainer
  // Should we deprecate it?
  const InlinedHashSet<NodeIndex>* node_to_execute_{nullptr};
#endif
  const bool single_thread_mode_;
};

using NotificationIndex = size_t;

void RunSince(size_t stream_idx, ExecutionContext& ctx, size_t since);
void ScheduleDownstream(ExecutionContext& ctx,
                        size_t trigger,
                        bool single_thread_mode);
}  // namespace onnxruntime
