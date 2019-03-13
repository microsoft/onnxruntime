// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <iostream>
#include <fstream>
#include <tuple>
#include <initializer_list>
#include "core/platform/ort_mutex.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

namespace profiling {

/**
 * Main class for profiling. It continues to accumulate events and produce
 * a corresponding "complete event (X)" in "chrome tracing" format.
 */
class Profiler {
 public:
  /// turned off by default.
  /// Even this function is marked as noexcept, the code inside it may throw exceptions
  Profiler() noexcept {};  //NOLINT

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

  bool FEnabled() const {
    return enabled_;
  }

  /*
  Record a single event. Time is measured till the call of this function from
  the start_time.
  */
  void EndTimeAndRecordEvent(EventCategory category,
                             const std::string& event_name,
                             TimePoint& start_time,
                             const std::initializer_list<std::pair<std::string, std::string>>& event_args = {},
                             bool sync_gpu = false);

  /*
  Write profile data to the given stream in chrome format defined below.
  https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#
  */
  std::string EndProfiling();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Profiler);

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
  static constexpr size_t max_num_events_ = 1000000;
  bool profile_with_logger_{false};
};

}  // namespace profiling
}  // namespace onnxruntime
