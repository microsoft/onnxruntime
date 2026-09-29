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
    // Note: it is not clear whether ORT should try to release the EP's profiler if the version is incorrect
    // since Release() was introduced in version 25 (along with the all profiling APIs).
    // if (profiler_impl.Release != nullptr) {
    //   profiler_impl.Release(&profiler_impl);
    // }

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

Status PluginEpProfiler::StartProfiling(TimePoint profiling_start_time) {
  // Store the epoch-based profiling start time for computing absolute correlation IDs in Start()/Stop().
  profiling_start_time_epoch_us_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(profiling_start_time.time_since_epoch()).count());

  // Compute the elapsed time since ORT's profiling start. This offset is epoch-independent.
  int64_t offset_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::high_resolution_clock::now() - profiling_start_time)
                          .count();

  Status status = ToStatusAndRelease(profiler_impl_.StartProfiling(&profiler_impl_, offset_ns));

  if (!status.IsOK()) {
    LOGS(logger_, ERROR) << "OrtEpProfilerImpl::StartProfiling() for " << ep_name_ << " returned an error OrtStatus: "
                         << status.ErrorMessage();
  }

  return status;
}

void PluginEpProfiler::EndProfiling(TimePoint /*profiling_start_time*/, profiling::Events& events) {
  OrtProfilingEventsContainer ep_events_container;
  Status status = ToStatusAndRelease(profiler_impl_.EndProfiling(&profiler_impl_,
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

void PluginEpProfiler::Start(uint64_t relative_ort_event_id) {
  if (profiler_impl_.StartEvent == nullptr) {
    return;
  }

  // Convert relative ORT event ID to an absolute correlation ID for the C API.
  // Because it is absolute rather than relative to profiling start, it is practically unique across concurrent
  // profiling sessions within the same process (collisions require sub-microsecond event concurrency) and can be
  // used directly as a correlation ID for EP profiling utilities (e.g., CUPTI or ROCTracer).
  uint64_t ort_event_correlation_id = relative_ort_event_id + profiling_start_time_epoch_us_;
  Status status = ToStatusAndRelease(profiler_impl_.StartEvent(&profiler_impl_, ort_event_correlation_id));
  if (!status.IsOK()) {
    // Log error but don't throw as profiling failures shouldn't break execution.
    LOGS(logger_, ERROR) << "OrtEpProfilerImpl::StartEvent() for " << ep_name_ << " returned an error OrtStatus: "
                         << status.ErrorMessage();
  }
}

void PluginEpProfiler::Stop(uint64_t relative_ort_event_id, const profiling::EventRecord& ort_event) {
  if (profiler_impl_.StopEvent == nullptr) {
    return;
  }

  // Convert relative ORT event ID to an absolute correlation ID for the C API.
  // Because it is absolute rather than relative to profiling start, it is practically unique across concurrent
  // profiling sessions within the same process (collisions require sub-microsecond event concurrency) and can be
  // used directly as a correlation ID for EP profiling utilities (e.g., CUPTI or ROCTracer).
  uint64_t ort_event_correlation_id = relative_ort_event_id + profiling_start_time_epoch_us_;
  Status status = ToStatusAndRelease(profiler_impl_.StopEvent(&profiler_impl_, ort_event_correlation_id,
                                                              ToOpaqueProfilingEvent(&ort_event)));
  if (!status.IsOK()) {
    // Log error but don't throw as profiling failures shouldn't break execution.
    LOGS(logger_, ERROR) << "OrtEpProfilerImpl::StopEvent() for " << ep_name_ << " returned an error OrtStatus: "
                         << status.ErrorMessage();
  }
}
}  // namespace onnxruntime
