// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../plugin_ep_utils.h"

/// <summary>
/// Example implementation of OrtEpProfilerImpl. ORT obtains an instance of this EP profiler by calling
/// OrtEp::CreateProfiler().
///
/// ORT calls the function pointers at appropriate times during a profiling session:
///   - StartProfiling once when profiling begins.
///   - [Optional] StartEvent / StopEvent around each ORT event (operator executions, session events, etc.).
///   - EndProfiling once when profiling ends to collect EP events.
///   - Release when ORT no longer needs the profiler.
/// </summary>
struct ExampleKernelEpProfiler : OrtEpProfilerImpl {
  const OrtEpApi& ep_api;
  int64_t ep_profiling_start_offset_ns_ = 0;
  std::chrono::high_resolution_clock::time_point ep_profiling_start_time_point_;
  uint64_t profiler_id = 0;

  explicit ExampleKernelEpProfiler(const OrtEpApi& api);
  ~ExampleKernelEpProfiler();

  static void ORT_API_CALL ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL StartProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                    int64_t ep_profiling_start_offset_ns) noexcept;

  static OrtStatus* ORT_API_CALL StartEventImpl(OrtEpProfilerImpl* this_ptr, uint64_t ort_event_correlation_id) noexcept;
  static OrtStatus* ORT_API_CALL StopEventImpl(OrtEpProfilerImpl* this_ptr, uint64_t ort_event_correlation_id,
                                               const OrtProfilingEvent* ort_event) noexcept;
  static OrtStatus* ORT_API_CALL EndProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                  OrtProfilingEventsContainer* events_container) noexcept;
};

/// <summary>
/// Singleton object that stores events from this EP's kernels and manages a stack of ORT event boundaries
/// used for annotating EP events with metadata from correlated ORT events.
///
/// This singleton maintains state per profiling session (i.e., per profiler). An OrtEpProfilerImpl must register
/// itself via RegisterProfiler().
///
/// An OrtEpProfilerImpl performs the following operations:
///   - RegisterProfiler()
///   - PushOrtEvent() / PopOrtEvent() as ORT provides StartEvent / StopEvent callbacks
///   - ConsumeEvents() to get all EP events when ORT calls OrtEpProfilerImpl::EndProfiling()
///
/// An EP kernel performs the following operations:
///   - AddEvent() to add a new EP event (e.g., kernel execution start and duration)
/// </summary>
class EpEventManager {
 public:
  struct Event {
    Event(std::string event_name, std::chrono::high_resolution_clock::time_point start_ts,
          std::chrono::high_resolution_clock::time_point end_ts)
        : name(std::move(event_name)), start_time(start_ts), end_time(end_ts), thread_id(std::this_thread::get_id()) {}

    std::string name;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::string ort_event_name;  // Set from the correlated ORT event
    std::thread::id thread_id;   // Thread that created this event
  };

  static EpEventManager& GetInstance();

  // Returns the active profiler ID for the current thread, or std::nullopt if no
  // ORT event is in progress on this thread. Use this from kernels to determine the correct
  // profiler ID for submitting profiling events during concurrent runs (each with its own run profiler).
  static std::optional<uint64_t> GetActiveProfilerId();

  uint64_t RegisterProfiler();
  void UnregisterProfiler(uint64_t profiler_id);

  void PushOrtEvent(uint64_t profiler_id);
  void PopOrtEvent(uint64_t profiler_id, const std::string& ort_event_name);

  void AddEpEvent(uint64_t profiler_id, Event event);

  void ConsumeEvents(uint64_t profiler_id, std::vector<Event>& events);

 private:
  struct ProfilerState {
    std::vector<Event> events;
  };

  mutable std::mutex mutex_;
  uint64_t next_profiler_id_{1};

  // profiler ID -> ProfilerState
  std::unordered_map<uint64_t, ProfilerState> profiler_state_;
};
