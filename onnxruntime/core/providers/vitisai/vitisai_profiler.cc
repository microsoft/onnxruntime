// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "vitisai_profiler.h"

#include "core/common/inlined_containers.h"

// Build marker for verifying component alignment
static const char* BUILD_MARKER_ORT_PROFILER = "[BUILD:ort_profiler:" __DATE__ " " __TIME__ "]";
static volatile const char* keep_ort_marker = BUILD_MARKER_ORT_PROFILER;


namespace onnxruntime {
namespace profiling {

#if defined(USE_VITISAI)

bool VitisaiProfiler::StartProfiling(TimePoint tp) {
  // Notify VAIP EP that profiling has started with base timestamp
  profiler_start(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     tp.time_since_epoch())
                     .count());
  return true;
}

void VitisaiProfiler::EndProfiling(TimePoint tp, Events& events) {
  // Notify VAIP EP that profiling has stopped
  profiler_stop();

  auto time_point =
      std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()).count();

  // Use v2 API - automatically falls back to v1 if vaip doesn't support v2
  std::vector<EventInfoV2> api_events;
  std::vector<EventInfoV2> kernel_events;
  profiler_collect_v2(api_events, kernel_events);

  for (auto& a : api_events) {
    // Use args from EventInfoV2 (6th element)
    events.emplace_back(EventCategory::API_EVENT,
                        std::get<1>(a),
                        std::get<2>(a),
                        std::get<0>(a),
                        std::get<3>(a) - time_point,
                        std::get<4>(a),
                        std::get<5>(a));
  }

  for (auto& k : kernel_events) {
    events.emplace_back(EventCategory::KERNEL_EVENT,
                        std::get<1>(k),
                        std::get<2>(k),
                        std::get<0>(k),
                        std::get<3>(k) - time_point,
                        std::get<4>(k),
                        std::get<5>(k));
  }
}

#endif

}  // namespace profiling
}  // namespace onnxruntime
