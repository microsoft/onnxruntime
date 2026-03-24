// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_event_profiling.h"

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"

namespace onnxruntime {

/*status*/
Status PluginEpProfiler::Create(OrtEpProfilerImpl& profiler_impl, const logging::Logger& logger,
                                const std::string& ep_name, std::unique_ptr<PluginEpProfiler>& profiler_out) {
  // plugin EP profiling APIs were introduced in ORT 1.25
  if (auto profiler_version = profiler_impl.ort_version_supported; profiler_version < 25) {
    if (profiler_impl.Release != nullptr) {
      profiler_impl.Release(&profiler_impl);
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "OrtEpProfilerImpl::ort_version_supported (", profiler_version, ") for ",
                           ep_name, " expected to be >= 25");
  }

  // Check presence of required OrtEpProfilerImpl functions
  if (profiler_impl.Release == nullptr ||
      profiler_impl.StartProfiling == nullptr ||
      profiler_impl.EndProfiling == nullptr) {
    if (profiler_impl.Release != nullptr) {
      profiler_impl.Release(&profiler_impl);
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "OrtEpProfilerImpl for ", ep_name, " is missing one or more ",
                           "required function implementations: Release, StartProfiling, and EndProfiling");
  }

  profiler_out = std::make_unique<PluginEpProfiler>(profiler_impl, logger, ep_name, PrivateTag{});
  return Status::OK();
}

PluginEpProfiler::PluginEpProfiler(OrtEpProfilerImpl& profiler_impl, const logging::Logger& logger,
                                   std::string ep_name, PrivateTag)
    : profiler_impl_{profiler_impl}, logger_{logger}, ep_name_(std::move(ep_name)) {}

PluginEpProfiler::~PluginEpProfiler() {
  profiler_impl_.Release(&profiler_impl_);
}

bool PluginEpProfiler::StartProfiling(TimePoint profiling_start_time) {
  int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   profiling_start_time.time_since_epoch())
                   .count();
  return profiler_impl_.StartProfiling(&profiler_impl_, ns);
}

void PluginEpProfiler::EndProfiling(TimePoint start_time, profiling::Events& events) {
  int64_t profiling_start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        start_time.time_since_epoch())
                                        .count();

  OrtProfilingEventsContainer ep_events_container;
  Status status = ToStatusAndRelease(profiler_impl_.EndProfiling(&profiler_impl_, profiling_start_time_ns,
                                                                 &ep_events_container));

  if (!status.IsOK()) {
    // Log error but don't throw as profiling failures shouldn't break execution.
    LOGS(logger_, ERROR) << "OrtEpProfilerImpl::EndProfiling() for " << ep_name_ << " returned an error OrtStatus: "
                         << status.ErrorMessage();
    return;
  }

  if (ep_events_container.events.empty()) {
    return;
  }

  // Append EP events to the overall events list.
  events.reserve(events.size() + ep_events_container.events.size());
  for (auto& record : ep_events_container.events) {
    events.emplace_back(std::move(record));
  }
}

void PluginEpProfiler::Start(uint64_t ort_event_id) {
  if (profiler_impl_.ort_version_supported >= 25 && profiler_impl_.StartEvent != nullptr) {
    profiler_impl_.StartEvent(&profiler_impl_, ort_event_id);
  }
}

void PluginEpProfiler::Stop(uint64_t ort_event_id, const profiling::EventRecord& ort_event) {
  if (profiler_impl_.ort_version_supported >= 25 && profiler_impl_.StopEvent != nullptr) {
    profiler_impl_.StopEvent(&profiler_impl_, ort_event_id, ToOpaqueProfilingEvent(&ort_event));
  }
}
}  // namespace onnxruntime
