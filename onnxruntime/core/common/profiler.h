// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <tuple>

#include "core/common/logging/logging.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

namespace profiling {

// uncomment the macro below, or use -DENABLE_STATIC_PROFILER_INSTANCE for debugging
// note that static profiler instance only works with single session
//#define ENABLE_STATIC_PROFILER_INSTANCE

/**
 * Main class for profiling. It continues to accumulate events and produce
 * a corresponding "complete event (X)" in "chrome tracing" format.
 */
class Profiler {
 public:
  /// turned off by default.
  /// Even this function is marked as noexcept, the code inside it may throw exceptions
  Profiler() noexcept {};  //NOLINT

  ~Profiler();

  /*
  Initializes Profiler with the session logger to log framework specific messages
  */
  void Initialize(const logging::Logger* session_logger);

  /*
  Send profiling data to custom logger
  */
  void StartProfiling(const logging::Logger* custom_logger);

  /*
  Start profiler and record beginning time.
  */
  template <typename T>
  void StartProfiling(const std::basic_string<T>& file_name);

  /*
  Produce current time point for any profiling action.
  */
  TimePoint StartTime() const;

  /*
  Whether data collection and output from this profiler is enabled.
  */
  bool IsEnabled() const {
    return enabled_;
  }
  /*
  Return the stored start time of profiler.
  On some platforms, this timer may not be as precise as nanoseconds
  For instance, on Windows and MacOS, the precision (high_resolution_clock) will be ~100ns
  */
  uint64_t GetStartTimeNs() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      profiling_start_time_.time_since_epoch()).count();
  }
  /*
  Record a single event. Time is measured till the call of this function from
  the start_time.
  */
  void EndTimeAndRecordEvent(EventCategory category,
                             const std::string& event_name,
                             const TimePoint& start_time,
                             const std::initializer_list<std::pair<std::string, std::string>>& event_args = {},
                             bool sync_gpu = false);

  /*
  Write profile data to the given stream in chrome format defined below.
  https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#
  */
  std::string EndProfiling();

  static Profiler& Instance() {
#ifdef ENABLE_STATIC_PROFILER_INSTANCE
    ORT_ENFORCE(instance_ != nullptr);
    return *instance_;
#else
    ORT_THROW("Static profiler instance is not enabled, please compile with -DENABLE_STATIC_PROFILER_INSTANCE");
#endif
  }

  /*
  Gets the maximum event count to set for new profiler instances.
  */
  static size_t GetGlobalMaxNumEvents() {
    return global_max_num_events_.load();
  }

  /*
  Sets the maximum event count to set for new profiler instances.
  Existing profiler instances will not be affected.
  */
  static void SetGlobalMaxNumEvents(size_t new_max_num_events) {
    global_max_num_events_.store(new_max_num_events);
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Profiler);

  /**
   * The maximum number of profiler records to collect.
   * This value is used to initialize the per-profiler maximum.
   * It can be set, but won't affect existing profilers.
   */
  static std::atomic<size_t> global_max_num_events_;

  // Mutex controlling access to profiler data
  OrtMutex mutex_;
  bool enabled_{false};
  std::ofstream profile_stream_;
  std::string profile_stream_file_;
  const logging::Logger* session_logger_{nullptr};
  const logging::Logger* custom_logger_{nullptr};
  TimePoint profiling_start_time_;
  std::vector<EventRecord> events_;
  bool max_events_reached{false};
  bool profile_with_logger_{false};
  const size_t max_num_events_{global_max_num_events_.load()};

#ifdef ENABLE_STATIC_PROFILER_INSTANCE
  static Profiler* instance_;
#endif
};

}  // namespace profiling
}  // namespace onnxruntime
