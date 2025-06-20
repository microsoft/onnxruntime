// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <string_view>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/profiler_common.h"
#include "core/providers/qnn/profiling_event_store.h"

namespace onnxruntime::qnn::profiling_utils {

/**
 * Traces a duration while in scope and records it to the specified `ProfilingEventStore`.
 *
 * Example usage:
 *   // record duration of my_api_call()
 *   {
 *      DurationTrace my_api_call_trace{GetProfilingEventStore(), profiling::API_EVENT, "my_api_call"};
 *      my_api_call();
 *   }
 */
class DurationTrace {
 public:
  [[nodiscard]] DurationTrace(ProfilingEventStore* profiling_event_store,
                              profiling::EventCategory category,
                              std::string_view name,
                              std::unordered_map<std::string, std::string> args = {})
      : profiling_event_store_{profiling_event_store},
        state_{} {
    if (profiling_event_store_ == nullptr) {
      return;
    }

    if (!profiling_event_store_->IsEnabled()) {
      return;
    }

    state_.emplace();

    state_->event = profiling::EventRecord{category,
                                           -1,  // process id
                                           -1,  // thread id
                                           std::string{name},
                                           -1,  // timestamp (populated later)
                                           -1,  // duration (populated later)
                                           std::move(args)};

    state_->start_time = TimePoint::clock::now();
  }

  ~DurationTrace() {
    if (!state_.has_value()) {
      return;
    }

    const auto end_time = TimePoint::clock::now();
    state_->event.ts = TimeDiffMicroSeconds(profiling_event_store_->BaseTime(), state_->start_time);
    state_->event.dur = TimeDiffMicroSeconds(state_->start_time, end_time);

    profiling_event_store_->AddEvent(std::move(state_->event));
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DurationTrace);

 private:
  struct DurationTraceState {
    profiling::EventRecord event;
    TimePoint start_time;
  };

  ProfilingEventStore* profiling_event_store_;
  std::optional<DurationTraceState> state_;
};

}  // namespace onnxruntime::qnn::profiling_utils
