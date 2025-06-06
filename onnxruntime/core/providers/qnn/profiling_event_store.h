// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <vector>
#include <mutex>

#include "core/common/profiler_common.h"

namespace onnxruntime::qnn {

/**
 * Manages a collection of profiling events. Events can be added or extracted.
 * The event store can be enabled or disabled. New events will only be added if it is enabled.
 */
class ProfilingEventStore {
 public:
  // Gets the base time point. Event timestamps should be relative to this time point.
  TimePoint BaseTime() const;

  bool IsEnabled() const;
  void SetEnabled(bool is_enabled);

  // Adds `event` to the event store if enabled. `event` will only be moved from if it is added.
  // Returns true if the event was added, false otherwise.
  bool AddEvent(profiling::EventRecord&& event);

  // Extracts all events from the event store.
  // `new_base_time` specifies a new time point that the event timestamps will be made relative to.
  std::vector<profiling::EventRecord> ExtractEvents(const TimePoint& new_base_time);

 private:
  const TimePoint base_time_{TimePoint::clock::now()};

  std::atomic<bool> is_enabled_{false};
  std::vector<profiling::EventRecord> profiling_events_{};
  std::mutex mutex_{};
};

}  // namespace onnxruntime::qnn
