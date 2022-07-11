#pragma once
#include "core/common/logging/logging.h"
#include "core/framework/ort_value.h"
#include "core/framework/iexecutor.h"
#include "core/framework/stream_handles.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {
class SessionState;
class IExecutionFrame;
class ExecutionFrame;

using OrtValueIndex = int;

struct ReleasePlan {
  std::unique_ptr<std::atomic_int[]> value_ref_counts_;
};

class DeviceStreamColloectionImpl;
class DeviceStreamColloection {
 public:
  DeviceStreamColloection(size_t num_streams);
  ~DeviceStreamColloection();
  void SetDeviceStream(size_t, std::unique_ptr<Stream> stream);
  void SetDeviceStream(size_t, Stream* stream);
  const std::vector<Stream*>& GetStreams() const;
  size_t NumStreams() const;

 private:
  std::unique_ptr<DeviceStreamColloectionImpl> impl_;
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

 private:
  std::atomic_int_fast32_t v_;
};

class SessionScope;

// execution context that support to execute a command on stream.
// The notifications got instantiated when execution context is constructed.
// TODO: if we merge the notifications to execution frame, we might don't need this.
class ExecutionContext {
public:
  ExecutionContext(const SessionState& sess_state,
                   int32_t num_streams,
                   std::vector<size_t> notification_owners,
                   const std::vector<int>& feed_mlvalue_idxs,
                   const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                   std::vector<OrtValue>& fetches,
                   const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                   size_t num_barriers,
                   const logging::Logger& sess_logger,
                   const DeviceStreamColloection& device_streams_map,
                   const bool& terminate_flag);

  const SessionState& GetSessionState() const;

  const logging::Logger& GetLogger() const;

  IExecutionFrame* GetExecutionFrame();

  synchronize::Notification* GetNotification(size_t idx);

  const bool& TerminateFlag() const;

  const Status& TaskStatus() const;

  bool DecCountDownBarrier(size_t barrier_id);

  Stream* GetDeviceStream(size_t idx);

  void CompleteTask();

  void WaitAll();

  void SetStatus(Status& status);

  void RecycleNodeInputs(onnxruntime::NodeIndex node_index);

  ~ExecutionContext();

  SessionScope* GetSessionScope() {
    return session_scope_;
  }

  void SetSessionScope(SessionScope* session_scope) {
    session_scope_ = session_scope;
  }

 private:
  const SessionState* session_state;
  std::unique_ptr<ExecutionFrame> frame;
  const logging::Logger* logger;
  std::vector<std::unique_ptr<synchronize::Notification>> notifications;
  std::unique_ptr<ReleasePlan> release_plan;
  const DeviceStreamColloection& device_stream_map_;
  std::vector<CountDownBarrier> count_down_barriers_;
  CountDownBarrier remain_tasks_;
  const bool& terminate_flag_;
  Status task_status_{Status::OK()};
  SessionScope* session_scope_{};
};

using NotificationIndex = size_t;

void RunSince(size_t stream_idx, ExecutionContext& ctx, size_t since);
void ScheduleDownstream(ExecutionContext& ctx, 
    onnxruntime::NotificationIndex notification_index,
    bool single_thread_mode);
}