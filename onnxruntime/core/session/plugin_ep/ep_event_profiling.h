// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/profiler_common.h"
#include "core/session/onnxruntime_c_api.h"

/// <summary>
/// Definition of the opaque OrtEpProfilingEvent type declared in the public C EP API.
/// An EP profiler creates instances via OrtEpApi::CreateEpProfilingEvent.
/// </summary>
struct OrtEpProfilingEvent {
  uint64_t ort_event_id;  // The ID of the ORT event to which this EP event is correlated.
  onnxruntime::profiling::EventRecord record;
};

// Definition of the opaque OrtEpProfilingEventsContainer type declared in the public C EP API.
// ORT creates an instance wrapping a profiling::Events vector and passes it to the EP's
// OrtEpProfilerImpl::EndProfiling() function.
// The EP calls OrtEpApi::EpProfilingEventsContainer_AddEvents to push events into this container.
struct OrtEpProfilingEventsContainer {
  std::vector<OrtEpProfilingEvent> ep_events;
};

namespace onnxruntime {
namespace logging {
class Logger;
}

/// <summary>
/// Wraps OrtEpProfilerImpl from a plugin EP into the C++ profiling::EpProfiler instance.
/// </summary>
class PluginEpProfiler final : public profiling::EpProfiler {
 public:
  explicit PluginEpProfiler(OrtEpProfilerImpl& profiler_impl, const logging::Logger& logger);
  ~PluginEpProfiler() override;

  PluginEpProfiler(const PluginEpProfiler&) = delete;
  PluginEpProfiler& operator=(const PluginEpProfiler&) = delete;

  bool StartProfiling(TimePoint profiling_start_time) override;
  void EndProfiling(TimePoint start_time, profiling::Events& events) override;

  void Start(uint64_t ort_event_id) override;
  void Stop(uint64_t ort_event_id) override;

 private:
  OrtEpProfilerImpl* profiler_impl_;
  const logging::Logger& logger_;
};

}  // namespace onnxruntime
