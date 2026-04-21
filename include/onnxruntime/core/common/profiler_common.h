// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/inlined_containers.h"

#include <string>

namespace onnxruntime {
namespace profiling {

// Profiling event categories.
// Note: Keep in sync with OrtProfilingEventCategory in onnxruntime_ep_c_api.h.
enum EventCategory {
  SESSION_EVENT = 0,
  NODE_EVENT,
  KERNEL_EVENT,
  API_EVENT,
  EVENT_CATEGORY_MAX
};

// Event descriptions for the above session events.
static constexpr const char* event_category_names_[EVENT_CATEGORY_MAX] = {
    "Session",
    "Node",
    "Kernel",
    "Api"};

// Timing record for all events.
struct EventRecord {
  EventRecord() = default;
  EventRecord(EventCategory category,
              int process_id,
              int thread_id,
              std::string&& event_name,
              long long time_stamp,
              long long duration,
              InlinedHashMap<std::string, std::string>&& event_args)
      : cat(category),
        pid(process_id),
        tid(thread_id),
        name(std::move(event_name)),
        ts(time_stamp),
        dur(duration),
        args(std::move(event_args)) {}

  EventRecord(EventCategory category,
              int process_id,
              int thread_id,
              const std::string& event_name,
              long long time_stamp,
              long long duration,
              const InlinedHashMap<std::string, std::string>& event_args)
      : cat(category),
        pid(process_id),
        tid(thread_id),
        name(event_name),
        ts(time_stamp),
        dur(duration),
        args(event_args) {}

  EventRecord(const EventRecord& other) = default;
  EventRecord(EventRecord&& other) noexcept = default;
  EventRecord& operator=(const EventRecord& other) = default;
  EventRecord& operator=(EventRecord&& other) = default;

  EventCategory cat = EventCategory::API_EVENT;
  int pid = -1;
  int tid = -1;
  std::string name{};
  long long ts = 0;
  long long dur = 0;
  InlinedHashMap<std::string, std::string> args{};
};

using Events = std::vector<EventRecord>;

// Execution Provider Profiler
class EpProfiler {
 public:
  virtual ~EpProfiler() = default;

  /// <summary>
  /// Called when profiling starts.
  /// Allows EP profiler to initialize profiling utilities and record the profiling start time.
  /// </summary>
  /// <param name="profiling_start_time">Timepoint denoting the start of profiling.</param>
  /// <returns>True if profiling was started successfully.</returns>
  virtual bool StartProfiling(TimePoint profiling_start_time) = 0;

  /// <summary>
  /// Called when profiling ends to collect the EP's new profiling events since the last call to StartProfiling.
  /// </summary>
  /// <param name="start_time">Timepoint denoting the start of profiling. Same value passed to StartProfiling.</param>
  /// <param name="events">Modifiable events container to which the EP profiler appends its events.</param>
  virtual void EndProfiling(TimePoint start_time, Events& events) = 0;

  /// <summary>
  /// Optional to override (default implementation does nothing).
  ///
  /// Called when an ORT event (e.g., session initialization, node kernel execution, etc.) starts.
  /// ORT pairs every Start call with a corresponding call to Stop with the same relative ORT event ID.
  /// EP profiler implementations may use the calls to Start and Stop to maintain a stack of ORT event IDs
  /// that can be correlated with EP events (e.g., GPU kernel events).
  ///
  /// A relative ORT event ID is computed as a timestamp offset relative to the profiling start time:
  ///     relative_ort_event_id =
  ///         std::chrono::duration_cast<std::chrono::microseconds>(event_start_time - profiling_start_time).count();
  ///
  /// Because relative ORT event IDs are relative to profiling start, different profiling sessions may reuse the same
  /// values. If the EP's profiling utilities (e.g., CUPTI or ROCTracer) require correlation IDs that are practically
  /// unique across concurrent profiling sessions (collisions require sub-microsecond event concurrency), then the
  /// EP profiler should compute an absolute correlation ID:
  ///     absolute_ort_correlation_id =
  ///        relative_ort_event_id +
  ///        std::chrono::duration_cast<std::chrono::microseconds>(profiling_start_time.time_since_epoch()).count();
  ///
  /// Note: For plugin EPs using the binary-stable C API (OrtEpProfilerImpl), ORT performs this conversion
  /// automatically. The C API's StartEvent/StopEvent receive the absolute correlation ID directly.
  /// </summary>
  /// <param name="relative_ort_event_id">
  /// Relative ID of the ORT event that is starting (microseconds since profiling start).
  /// The same value is passed to a corresponding call to Stop.
  /// </param>
  virtual void Start(uint64_t /*relative_ort_event_id*/) {}

  /// <summary>
  /// Optional to override (default implementation does nothing).
  ///
  /// Called when an ORT event (e.g., session initialization, node kernel execution, etc.) ends.
  /// ORT pairs every Start call with a corresponding call to Stop with the same relative ORT event ID.
  /// EP profiler implementations may use the calls to Start and Stop to maintain a stack of ORT event IDs
  /// that can be correlated with EP events (e.g., GPU kernel events).
  ///
  /// The ort_event parameter provides the full ORT event record including metadata such as op_name
  /// (in event args), event name, category, timestamps, etc. EP profilers can use this to annotate
  /// their own events with ORT event context.
  /// </summary>
  /// <param name="relative_ort_event_id">
  /// Relative ID of the ORT event that is ending (microseconds since profiling start).
  /// The same value was passed to a corresponding call to Start.
  /// </param>
  /// <param name="ort_event">
  /// The ORT event record containing metadata for this event.
  /// </param>
  virtual void Stop(uint64_t /*relative_ort_event_id*/, const EventRecord& /*ort_event*/) {}
};

// Demangle C++ symbols
std::string demangle(const char* name);
std::string demangle(const std::string& name);

}  // namespace profiling
}  // namespace onnxruntime
