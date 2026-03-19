// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_event_profiling.h"

#include <map>
#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"

namespace onnxruntime {

// Merge EP-provided events into the existing event list by timestamp.
// EP events with an ort_event_id
// For each EP event whose ort_event_id matches an existing event's timestamp, inherits
// op_name and parent_name from that parent event. Events are interleaved
// in timestamp order. This mirrors GPUProfilerBase::MergeEvents from
// gpu_profiler_common.h.
static void MergeEpEvents(std::map<uint64_t, profiling::Events>& ep_events, profiling::Events& events) {
  profiling::Events merged_events;
  merged_events.reserve(events.size() + ep_events.size());

  auto event_iter = events.begin();
  auto event_end = events.end();

  for (auto& [ort_event_id, ep_group] : ep_events) {
    if (ep_group.empty()) {
      continue;
    }

    // TODO(adrianlizarraga): Handle uncorrelated EP events correctly

    // An ORT event ID is just the timestamp of an ORT event in microseconds, relative to the
    // profiling start time. So, we can use it to match the parent ORT event.
    long long parent_ts_to_match = static_cast<long long>(ort_event_id);

    // Advance past existing events with earlier timestamps,
    // and past all-but-the-last existing event sharing this timestamp.
    while (event_iter != event_end &&
           (event_iter->ts < parent_ts_to_match ||
            (event_iter->ts == parent_ts_to_match &&
             (event_iter + 1) != event_end &&
             (event_iter + 1)->ts == parent_ts_to_match))) {
      merged_events.emplace_back(std::move(*event_iter));
      ++event_iter;
    }

    bool found_parent_ort_event = false;
    std::string op_name;
    std::string parent_name;

    if (event_iter != event_end && event_iter->ts == parent_ts_to_match) {
      // Found a matching parent event. Get its op_name and name.
      found_parent_ort_event = true;
      auto op_name_it = event_iter->args.find("op_name");
      if (op_name_it != event_iter->args.end()) {
        op_name = op_name_it->second;
      }
      parent_name = event_iter->name;
      merged_events.emplace_back(std::move(*event_iter));
      ++event_iter;
    }

    if (found_parent_ort_event) {
      for (auto& evt : ep_group) {
        evt.args["op_name"] = op_name;
        evt.args["parent_name"] = parent_name;
      }
    }

    merged_events.insert(merged_events.end(),
                         std::make_move_iterator(ep_group.begin()),
                         std::make_move_iterator(ep_group.end()));
  }

  // Move any remaining existing events
  merged_events.insert(merged_events.end(),
                       std::make_move_iterator(event_iter),
                       std::make_move_iterator(event_end));

  std::swap(events, merged_events);
}

PluginEpProfiler::PluginEpProfiler(OrtEpProfilerImpl& profiler_impl, const logging::Logger& logger)
    : profiler_impl_{&profiler_impl}, logger_{logger} {}

PluginEpProfiler::~PluginEpProfiler() {
  if (profiler_impl_ != nullptr) {
    profiler_impl_->Release(profiler_impl_);
  }
}

bool PluginEpProfiler::StartProfiling(TimePoint profiling_start_time) {
  int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   profiling_start_time.time_since_epoch())
                   .count();
  return profiler_impl_->StartProfiling(profiler_impl_, ns);
}

void PluginEpProfiler::EndProfiling(TimePoint start_time, profiling::Events& events) {
  int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   start_time.time_since_epoch())
                   .count();

  // Collect EP events into a separate buffer so we can merge them into the
  // existing events afterward (matching timestamps and annotating with
  // parent op_name/parent_name, mirroring GPUProfilerBase::MergeEvents).
  OrtEpProfilingEventsContainer container;
  Status status = ToStatusAndRelease(profiler_impl_->EndProfiling(profiler_impl_, ns, &container));
  if (!status.IsOK()) {
    // Log error but don't throw as profiling failures shouldn't break execution.
    LOGS(logger_, ERROR) << "OrtEpProfilerImpl::EndProfiling() returned an error OrtStatus: "
                         << status.ErrorMessage();
    return;
  }

  if (container.ep_events.empty()) {
    return;
  }

  // Group EP events by ort_event_id
  std::map<uint64_t, profiling::Events> grouped_ep_events = [](OrtEpProfilingEventsContainer& container) {
    std::map<uint64_t, profiling::Events> result;

    for (OrtEpProfilingEvent& ep_event : container.ep_events) {
      result[ep_event.ort_event_id].emplace_back(std::move(ep_event.record));
    }

    container.ep_events.clear();  // container has been gutted, so clear it to prevent accidental use.
    return result;
  }(container);

  MergeEpEvents(grouped_ep_events, events);
}

void PluginEpProfiler::Start(uint64_t ort_event_id) {
  if (profiler_impl_->ort_version_supported >= 25 && profiler_impl_->StartEvent != nullptr) {
    profiler_impl_->StartEvent(profiler_impl_, ort_event_id);
  }
}

void PluginEpProfiler::Stop(uint64_t ort_event_id) {
  if (profiler_impl_->ort_version_supported >= 25 && profiler_impl_->StopEvent != nullptr) {
    profiler_impl_->StopEvent(profiler_impl_, ort_event_id);
  }
}
}  // namespace onnxruntime
