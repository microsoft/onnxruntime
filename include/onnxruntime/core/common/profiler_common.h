// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

namespace onnxruntime {
namespace profiling {

enum EventCategory {
  SESSION_EVENT = 0,
  NODE_EVENT,
  KERNEL_EVENT,
  EVENT_CATEGORY_MAX
};

// Event descriptions for the above session events.
static constexpr const char* event_categor_names_[EVENT_CATEGORY_MAX] = {
    "Session",
    "Node",
    "Kernel"};

// Timing record for all events.
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

class EpProfiler {
 public:
  virtual ~EpProfiler() = default;
  virtual bool StartProfiling() = 0;
  virtual std::vector<EventRecord> StopProfiling() = 0;
};

}  //namespace profiling
}  //namespace onnxruntime