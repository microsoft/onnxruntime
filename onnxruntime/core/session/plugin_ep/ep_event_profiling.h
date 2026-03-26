// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/profiler_common.h"
#include "core/session/onnxruntime_c_api.h"

// OrtProfilingEvent is an opaque C alias for profiling::EventRecord.
// The C API forward-declares it as an incomplete type. Internally, we convert between
// the two via reinterpret_cast using the helpers below.

inline const OrtProfilingEvent* ToOpaqueProfilingEvent(const onnxruntime::profiling::EventRecord* r) {
  return reinterpret_cast<const OrtProfilingEvent*>(r);
}
inline OrtProfilingEvent* ToOpaqueProfilingEvent(onnxruntime::profiling::EventRecord* r) {
  return reinterpret_cast<OrtProfilingEvent*>(r);
}
inline const onnxruntime::profiling::EventRecord* FromOpaqueProfilingEvent(const OrtProfilingEvent* e) {
  return reinterpret_cast<const onnxruntime::profiling::EventRecord*>(e);
}
inline onnxruntime::profiling::EventRecord* FromOpaqueProfilingEvent(OrtProfilingEvent* e) {
  return reinterpret_cast<onnxruntime::profiling::EventRecord*>(e);
}

// Definition of the opaque OrtProfilingEventsContainer type declared in the public C EP API.
// ORT creates an instance wrapping a profiling::Events vector and passes it to the EP's
// OrtEpProfilerImpl::EndProfiling() function.
// The EP calls OrtEpApi::ProfilingEventsContainer_AddEvents to push events into this container.
struct OrtProfilingEventsContainer {
  onnxruntime::profiling::Events events;
};

namespace onnxruntime {
namespace logging {
class Logger;
}

/// <summary>
/// Wraps OrtEpProfilerImpl from a plugin EP into the C++ profiling::EpProfiler instance.
/// </summary>
class PluginEpProfiler final : public profiling::EpProfiler {
 private:
  struct PrivateTag {};

 public:
  static Status Create(OrtEpProfilerImpl& profiler_impl, const logging::Logger& logger,
                       const std::string& ep_name, std::unique_ptr<PluginEpProfiler>& profiler_out);

  // Do not use constructor. Use PluginEpProfiler::Create() for validation.
  PluginEpProfiler(OrtEpProfilerImpl& profiler_impl, const logging::Logger& logger, std::string ep_name, PrivateTag);
  ~PluginEpProfiler() override;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PluginEpProfiler);

  bool StartProfiling(TimePoint profiling_start_time) override;
  void EndProfiling(TimePoint start_time, profiling::Events& events) override;

  void Start(uint64_t relative_ort_event_id) override;
  void Stop(uint64_t relative_ort_event_id, const profiling::EventRecord& ort_event) override;

 private:
  OrtEpProfilerImpl& profiler_impl_;
  const logging::Logger& logger_;
  std::string ep_name_;
  uint64_t profiling_start_time_epoch_us_{0};
};

}  // namespace onnxruntime
