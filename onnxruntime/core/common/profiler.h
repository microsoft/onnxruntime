// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <iostream>
#include <fstream>
#include "core/common/logging/logging.h"

namespace onnxruntime {

namespace profiling {

enum EventCategory {
  SESSION_EVENT = 0,
  NODE_EVENT,
  EVENT_CATEGORY_MAX
};

/*
Event descriptions for the above session events.
*/
static constexpr const char* event_categor_names_[EVENT_CATEGORY_MAX] = {
    "Session",
    "Node"};

/*
Timing record for all events.
*/
struct EventRecord {
  EventRecord(EventCategory category,
              int process_id,
              int thread_id,
              std::string event_name,
              long long time_stamp,
              long long duration,
              std::unordered_map<std::string, std::string>&& event_args) : cat(category),
                                                                           pid(process_id),
                                                                           tid(thread_id),
                                                                           name(std::move(event_name)),
                                                                           ts(time_stamp),
                                                                           dur(duration),
                                                                           args(event_args) {}
  EventCategory cat;
  int pid;
  int tid;
  std::string name;
  long long ts;
  long long dur;
  std::unordered_map<std::string, std::string> args;
};

/*
Main class for profiling. It continues to accumulate events and produce
a corresponding "complete event (X)" in "chrome tracing" format.
*/
class Profiler {
 public:
  Profiler() noexcept {};  // turned off by default.

  /*
  Start profiler and record beginning time.
  */
  void StartProfiling(const logging::Logger* session_logger, const std::string& file_name);

  /*
  Produce current time point for any profiling action.
  */
  TimePoint StartTime() const;

  /*
  Record a single event. Time is measured till the call of this function from
  the start_time.
  */
  void EndTimeAndRecordEvent(EventCategory category,
                             const std::string& event_name,
                             TimePoint& start_time,
                             std::unordered_map<std::string, std::string>&& event_args = std::unordered_map<std::string, std::string>(),
                             bool sync_gpu = false);

  /*
  Write profile data to the given stream in chrome format defined below.
  https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#
  */
  std::string WriteProfileData();

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Profiler);

  // Mutex controlling access to profiler data
  std::mutex mutex_;
  bool enabled_{false};
  std::ofstream profile_stream_;
  std::string profile_stream_file_;
  const logging::Logger* session_logger_{nullptr};
  TimePoint profiling_start_time_;
  std::vector<EventRecord> events_;
  bool max_events_reached{false};
  static constexpr size_t max_num_events_ = 1000000;
};

}  // namespace profiling
}  // namespace onnxruntime
