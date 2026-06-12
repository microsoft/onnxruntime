// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_profiler_plugin.h"

#if defined(ENABLE_CUDA_PROFILING)

#include <map>
#include <string>
#include <vector>

namespace onnxruntime {
namespace cuda_plugin {

CudaPluginEpProfiler::CudaPluginEpProfiler(const OrtEpApi& api)
    : OrtEpProfilerImpl{}, ep_api(api) {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  StartProfiling = StartProfilingImpl;
  EndProfiling = EndProfilingImpl;
  StartEvent = StartEventImpl;
  StopEvent = StopEventImpl;

  auto& manager = profiling::CUPTIManager::GetInstance();
  client_handle_ = manager.RegisterClient();
}

CudaPluginEpProfiler::~CudaPluginEpProfiler() {
  auto& manager = profiling::CUPTIManager::GetInstance();
  manager.DeregisterClient(client_handle_);
}

/*static*/
void ORT_API_CALL CudaPluginEpProfiler::ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept {
  delete static_cast<CudaPluginEpProfiler*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL CudaPluginEpProfiler::StartProfilingImpl(
    OrtEpProfilerImpl* this_ptr,
    int64_t ep_profiling_start_offset_ns) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  auto* self = static_cast<CudaPluginEpProfiler*>(this_ptr);

  auto now = TimePoint::clock::now();

  // Reconstruct the approximate ORT profiling start time so that GPU event
  // timestamps (computed by CUPTIManager::Consume) are relative to ORT's start.
  // The result equals (ORT's profiling start) + (cross-DLL call latency), which
  // is typically < 1 µs — acceptable for profiling-level accuracy.
  self->ort_profiling_start_ = now -
                               std::chrono::duration_cast<TimePoint::duration>(
                                   std::chrono::nanoseconds(ep_profiling_start_offset_ns));

  auto& manager = profiling::CUPTIManager::GetInstance();
  manager.StartLogging();

  if (!manager.IsTracingEnabled()) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL,
        "CUPTI activity tracing failed to start. "
        "GPU kernel events will not be available in the profile. "
        "Check that the CUDA driver supports CUPTI and the CUPTI library is accessible.");
  }

  return nullptr;
  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaPluginEpProfiler::StartEventImpl(
    OrtEpProfilerImpl* this_ptr,
    uint64_t ort_event_correlation_id) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  auto* self = static_cast<CudaPluginEpProfiler*>(this_ptr);

  // The bridge provides an absolute epoch-based correlation ID. Pass TimePoint{}
  // (epoch) so PushCorrelation adds zero offset and the unique_cid equals the
  // correlation ID directly. This avoids double-adding the epoch offset that
  // GPUTracerManager::PushCorrelation normally computes.
  auto& manager = profiling::CUPTIManager::GetInstance();
  manager.PushCorrelation(self->client_handle_, ort_event_correlation_id, TimePoint{});

  return nullptr;
  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaPluginEpProfiler::StopEventImpl(
    OrtEpProfilerImpl* /*this_ptr*/,
    uint64_t /*ort_event_correlation_id*/,
    const OrtProfilingEvent* /*ort_event*/) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto& manager = profiling::CUPTIManager::GetInstance();
  manager.PopCorrelation();

  return nullptr;
  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaPluginEpProfiler::EndProfilingImpl(
    OrtEpProfilerImpl* this_ptr,
    OrtProfilingEventsContainer* c_events_container) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  auto* self = static_cast<CudaPluginEpProfiler*>(this_ptr);

  auto& manager = profiling::CUPTIManager::GetInstance();

  // Consume GPU events. Timestamps are computed relative to ort_profiling_start_
  // by CUPTIManager::ProcessActivityBuffers, so they match ORT's timeline.
  std::map<uint64_t, profiling::Events> event_map;
  manager.Consume(self->client_handle_, self->ort_profiling_start_, event_map);

  // Flatten all GPU events and convert to OrtProfilingEvent.
  std::vector<Ort::ProfilingEvent> events;
  for (auto& kv : event_map) {
    auto& event_list = kv.second;
    for (const auto& record : event_list) {
      // Build parallel key/value arrays to use the raw-pointer ProfilingEvent
      // constructor, avoiding a copy from InlinedHashMap to std::unordered_map.
      InlinedVector<const char*> arg_keys;
      InlinedVector<const char*> arg_values;
      arg_keys.reserve(record.args.size());
      arg_values.reserve(record.args.size());
      for (const auto& [k, v] : record.args) {
        arg_keys.push_back(k.c_str());
        arg_values.push_back(v.c_str());
      }

      events.emplace_back(
          OrtProfilingEventCategory_KERNEL,
          record.pid,
          record.tid,
          record.name.c_str(),
          record.ts,
          record.dur,
          arg_keys.data(),
          arg_values.data(),
          arg_keys.size());
    }
  }

  if (!events.empty()) {
    Ort::UnownedProfilingEventsContainer events_container(c_events_container);
    Ort::Status status = events_container.AddEvents(events);
    return status.release();
  }

  return nullptr;
  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime

#endif  // defined(ENABLE_CUDA_PROFILING)
