// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "profiler.h"

#include <chrono>
#include <utility>
#include <vector>

#include <gsl/gsl>

namespace onnxruntime::webgpu::ep {

using onnxruntime::ep::Api;

Profiler::Profiler(std::unique_ptr<profiling::EpProfiler> impl, bool& profiling_active)
    : OrtEpProfilerImpl{}, impl_{std::move(impl)}, profiling_active_{profiling_active} {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  StartProfiling = StartProfilingImpl;
  EndProfiling = EndProfilingImpl;
}

void ORT_API_CALL Profiler::ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept {
  auto* self = static_cast<Profiler*>(this_ptr);
  // Input validation can fail after StartProfiling without a matching EndProfiling.
  if (self->profiling_started_) {
    self->profiling_active_ = false;
  }
  delete self;
}

OrtStatus* ORT_API_CALL Profiler::StartProfilingImpl(
    OrtEpProfilerImpl* this_ptr, int64_t profiling_start_offset_ns) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto& self = *static_cast<Profiler*>(this_ptr);
  // Reconstruct ORT's start on the plugin's clock instead of sharing C++ clock objects across the ABI.
  self.profiling_start_time_ = TimePoint::clock::now() -
                               std::chrono::duration_cast<TimePoint::duration>(
                                   std::chrono::nanoseconds(profiling_start_offset_ns));
  const auto status = self.impl_->StartProfiling(self.profiling_start_time_);
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str());
  }
  self.profiling_active_ = true;
  self.profiling_started_ = true;
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Profiler::EndProfilingImpl(
    OrtEpProfilerImpl* this_ptr, OrtProfilingEventsContainer* events_container) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto& self = *static_cast<Profiler*>(this_ptr);
  auto stop_profiling = gsl::finally([&self] {
    self.profiling_active_ = false;
    self.profiling_started_ = false;
  });
  profiling::Events records;
  self.impl_->EndProfiling(self.profiling_start_time_, records);
  if (records.empty()) {
    return nullptr;
  }

  std::vector<Ort::ProfilingEvent> events;
  events.reserve(records.size());
  for (const auto& record : records) {
    InlinedVector<const char*> keys;
    InlinedVector<const char*> values;
    keys.reserve(record.args.size());
    values.reserve(record.args.size());
    for (const auto& [key, value] : record.args) {
      keys.push_back(key.c_str());
      values.push_back(value.c_str());
    }
    events.emplace_back(static_cast<OrtProfilingEventCategory>(record.cat),
                        record.pid, record.tid, record.name.c_str(), record.ts, record.dur,
                        keys.data(), values.data(), keys.size());
  }
  Ort::UnownedProfilingEventsContainer container{events_container};
  return container.AddEvents(events).release();
  EXCEPTION_TO_RETURNED_STATUS_END
}

}  // namespace onnxruntime::webgpu::ep
