// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../plugin_ep_utils.h"

/// <summary>
/// Example implementation of OrtEpProfilerImpl. ORT obtains an instance of this EP profiler by calling
/// OrtEp::GetProfiler().
///
/// ORT calls the function pointers at appropriate times during a profiling session:
///   - StartProfiling once when profiling begins.
///   - [Optional] StartEvent / StopEvent around each ORT event (operator executions, session events, etc.).
///   - EndProfiling once when profiling ends to collect EP events.
///   - Release when ORT no longer needs the profiler.
/// </summary>
struct ExampleKernelEpProfiler : OrtEpProfilerImpl {
  const OrtEpApi& ep_api;
  int64_t profiling_start_time_ns = 0;
  uint64_t client_id = 0;

  explicit ExampleKernelEpProfiler(const OrtEpApi& api);
  ~ExampleKernelEpProfiler();

  static void ORT_API_CALL ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept;
  static bool ORT_API_CALL StartProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                              int64_t profiling_start_time_ns) noexcept;

  static void ORT_API_CALL StartEventImpl(OrtEpProfilerImpl* this_ptr, uint64_t ort_event_id) noexcept;
  static void ORT_API_CALL StopEventImpl(OrtEpProfilerImpl* this_ptr, uint64_t ort_event_id) noexcept;
  static OrtStatus* ORT_API_CALL EndProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                  int64_t profiling_start_time_ns,
                                                  OrtEpProfilingEventsContainer* events_container) noexcept;
};

/// <summary>
/// Singleton object that stores events from this EP's kernels and manages the stack of ORT event IDs
/// used for correlating EP events with ORT events.
///
/// This singleton maintains state per profiling session (i.e., a client). An OrtEpProfilerImpl must register itself
/// as a client via RegisterClient().
///
/// An OrtEpProfilerImpl performs the following operations:
///   - RegisterClient()
///   - PushOrtEventId() / PopOrtEventId() as ORT provides events
///   - ConsumeEvents() to get all EP events when ORT calls OrtEpProfilerImpl::EndProfiling()
///
/// An EP kernel performs the following operations:
///   - PeekOrtEventId() to get the ID of the ORT event with which an EP event is correlated.
///   - AddEvent() to add a new EP event (e.g., kernel execution start and duration)
/// </summary>
class EpEventManager {
 public:
  struct Event {
    std::string name;
    uint64_t ort_event_id;
    int64_t ts_ns;
    int64_t dur_ns;
  };

  static EpEventManager& GetInstance();

  uint64_t RegisterClient();
  void UnregisterClient(uint64_t client_id);

  void StartProfiling();

  uint64_t PeekOrtEventId(uint64_t client_id);
  void PushOrtEventId(uint64_t client_id, uint64_t ort_event_id);
  void PopOrtEventId(uint64_t client_id);

  void AddEpEvent(uint64_t client_id, Event event);

  void ConsumeEvents(uint64_t client_id, std::vector<Event>& events);

 private:
  // Caller should hold mutex_
  void Shutdown();

  struct ClientState {
    std::vector<uint64_t> ort_event_id_stack;
    std::vector<Event> events;
  };

  mutable std::mutex mutex_;
  uint64_t next_client_id_;
  uint64_t num_clients_;
  bool enabled_;

  // client ID -> ClientState
  std::unordered_map<uint64_t, ClientState> client_state_;
};
