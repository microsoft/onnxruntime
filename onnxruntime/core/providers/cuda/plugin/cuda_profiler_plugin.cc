// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_profiler_plugin.h"

#if defined(ENABLE_CUDA_PROFILING)

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
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
    OrtEpProfilerImpl* this_ptr,
    uint64_t ort_event_correlation_id,
    const OrtProfilingEvent* ort_event) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  auto* self = static_cast<CudaPluginEpProfiler*>(this_ptr);

  // Always pop the CUPTI external correlation push performed in StartEvent,
  // regardless of category — even if metadata extraction below partially fails.
  auto& manager = profiling::CUPTIManager::GetInstance();
  manager.PopCorrelation();

  // For NODE_EVENT events, capture the originating node's identity now so that
  // EndProfiling can annotate the GPU kernel/memcpy events produced under this
  // correlation ID. Accessor failures are non-fatal: we simply skip annotation
  // for this event and rely on ort_correlation_id alone for linkage.
  if (ort_event != nullptr) {
    const auto& api = self->ep_api;

    OrtProfilingEventCategory category = OrtProfilingEventCategory_KERNEL;
    if (OrtStatus* s = api.ProfilingEvent_GetCategory(ort_event, &category); s != nullptr) {
      Ort::GetApi().ReleaseStatus(s);
      return nullptr;
    }

    if (category == OrtProfilingEventCategory_NODE) {
      OrtNodeInfo info;

      const char* event_name = nullptr;
      if (OrtStatus* s = api.ProfilingEvent_GetName(ort_event, &event_name); s != nullptr) {
        Ort::GetApi().ReleaseStatus(s);
      } else if (event_name != nullptr) {
        info.event_name = event_name;
      }

      const char* op_name = nullptr;
      if (OrtStatus* s = api.ProfilingEvent_GetArgValue(ort_event, "op_name", &op_name); s != nullptr) {
        Ort::GetApi().ReleaseStatus(s);
      } else if (op_name != nullptr) {
        info.op_name = op_name;
      }

      const char* node_index = nullptr;
      if (OrtStatus* s = api.ProfilingEvent_GetArgValue(ort_event, "node_index", &node_index); s != nullptr) {
        Ort::GetApi().ReleaseStatus(s);
      } else if (node_index != nullptr) {
        info.node_index = node_index;
      }

      std::lock_guard<std::mutex> lock(self->node_info_mutex_);
      self->correlation_to_node_[ort_event_correlation_id] = std::move(info);
    }
  }

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

  // Snapshot the correlation→node map under lock and clear it; subsequent
  // lookups can then run lock-free for the duration of event flattening.
  std::unordered_map<uint64_t, OrtNodeInfo> node_info;
  {
    std::lock_guard<std::mutex> lock(self->node_info_mutex_);
    node_info.swap(self->correlation_to_node_);
  }

  // Flatten all GPU events and convert to OrtProfilingEvent.
  std::vector<Ort::ProfilingEvent> events;
  for (auto& kv : event_map) {
    const uint64_t correlation_id = kv.first;
    auto& event_list = kv.second;

    // Resolve ORT-side attribution for this correlation ID (if any).
    const OrtNodeInfo* info = nullptr;
    if (auto it = node_info.find(correlation_id); it != node_info.end()) {
      info = &it->second;
    }

    // Stringify correlation ID once per outer iteration; storage must outlive
    // every Ort::ProfilingEvent constructor call below. The constructor copies
    // these strings into the container (see ProfilingEventsContainer_AddEvents),
    // so per-record local storage would also work, but lifting it here avoids
    // redundant work.
    const std::string correlation_id_str = std::to_string(correlation_id);

    for (const auto& record : event_list) {
      // Build parallel key/value arrays to use the raw-pointer ProfilingEvent
      // constructor, avoiding a copy from InlinedHashMap to std::unordered_map.
      // Reserve enough headroom for the CUPTI args plus up to 4 ORT annotations
      // (ort_correlation_id always; ort_event_name / ort_op_name / ort_node_index
      // when ORT-side metadata is available).
      InlinedVector<const char*> arg_keys;
      InlinedVector<const char*> arg_values;
      arg_keys.reserve(record.args.size() + 4);
      arg_values.reserve(record.args.size() + 4);
      for (const auto& [k, v] : record.args) {
        arg_keys.push_back(k.c_str());
        arg_values.push_back(v.c_str());
      }

      // Always emit ort_correlation_id so consumers can join GPU events back
      // to ORT events even when per-node attribution wasn't captured (e.g. the
      // event came from a non-NODE category, or StopEvent ran before the GPU
      // activity was finalized).
      arg_keys.push_back("ort_correlation_id");
      arg_values.push_back(correlation_id_str.c_str());

      if (info != nullptr) {
        if (!info->event_name.empty()) {
          arg_keys.push_back("ort_event_name");
          arg_values.push_back(info->event_name.c_str());
        }
        if (!info->op_name.empty()) {
          arg_keys.push_back("ort_op_name");
          arg_values.push_back(info->op_name.c_str());
        }
        if (!info->node_index.empty()) {
          arg_keys.push_back("ort_node_index");
          arg_values.push_back(info->node_index.c_str());
        }
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
