// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/profiling_event_store.h"

namespace onnxruntime::qnn {

TimePoint ProfilingEventStore::BaseTime() const {
  return base_time_;
}

bool ProfilingEventStore::IsEnabled() const {
  return is_enabled_;
}

void ProfilingEventStore::SetEnabled(bool is_enabled) {
  std::scoped_lock g{mutex_};
  is_enabled_ = is_enabled;
}

bool ProfilingEventStore::AddEvent(profiling::EventRecord&& event) {
  // check whether the store is enabled first without locking
  if (!IsEnabled()) {
    return false;
  }

  std::scoped_lock g{mutex_};

  if (!is_enabled_) {
    return false;
  }

  profiling_events_.emplace_back(std::move(event));
  return true;
}

std::vector<profiling::EventRecord> ProfilingEventStore::ExtractEvents(const TimePoint& new_base_time) {
  std::vector<profiling::EventRecord> extracted_profiling_events{};
  {
    std::scoped_lock g{mutex_};
    extracted_profiling_events.swap(profiling_events_);
  }

  // adjust timestamps for different base times
  const auto base_time_offset_us =
      std::chrono::duration_cast<std::chrono::microseconds>(new_base_time - base_time_).count();

  for (auto& event : extracted_profiling_events) {
    event.ts -= base_time_offset_us;
  }

  return extracted_profiling_events;
}

}  // namespace onnxruntime::qnn
