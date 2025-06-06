// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/qnn_profiler.h"

#include "core/providers/qnn/profiling_event_store.h"

namespace onnxruntime::qnn {

QnnProfiler::QnnProfiler(std::shared_ptr<ProfilingEventStore> event_store)
    : event_store_{std::move(event_store)} {
}

bool QnnProfiler::StartProfiling(TimePoint /*start_time*/) {
  event_store_->SetEnabled(true);
  return true;
}

void QnnProfiler::EndProfiling(TimePoint start_time, profiling::Events& events) {
  event_store_->SetEnabled(false);

  auto extracted_events = event_store_->ExtractEvents(start_time);
  events.insert(events.end(),
                std::make_move_iterator(extracted_events.begin()),
                std::make_move_iterator(extracted_events.end()));
}

}  // namespace onnxruntime::qnn
