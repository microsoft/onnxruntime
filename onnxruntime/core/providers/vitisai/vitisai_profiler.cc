// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "vitisai_profiler.h"

#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace profiling {

#if defined(USE_VITISAI)

bool VitisaiProfiler::StartProfiling(TimePoint tp) {
  // Notify VAIP EP that profiling has started with base timestamp
  profiler_start(std::chrono::duration_cast<std::chrono::microseconds>(
                     tp.time_since_epoch())
                     .count());
  return true;
}

void VitisaiProfiler::EndProfiling(TimePoint tp, Events& events) {
  // Notify VAIP EP that profiling has stopped
  // Contract: the VAIP EP must preserve any pending/buffered event data after
  // profiler_stop() returns, so that the subsequent profiler_collect[_v2]()
  // call below can still retrieve the full set of recorded events. The EP is
  // expected to release that buffered data only after collection completes
  // (or on the next profiler_start()).
  profiler_stop();

  auto time_point =
      std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()).count();

  // Try v2 first
  // Use the free function wrappers which internally null-check the pointers.
  std::vector<EventInfoV2> api_events_v2;
  std::vector<EventInfoV2> kernel_events_v2;
  const bool v2_available = profiler_collect_v2(api_events_v2, kernel_events_v2);

  if (v2_available) {
    for (auto& a : api_events_v2) {
      auto& src_args = std::get<5>(a);
      decltype(EventRecord::args) args(src_args.begin(), src_args.end());
      events.emplace_back(EventCategory::API_EVENT,
                          std::get<1>(a),
                          std::get<2>(a),
                          std::get<0>(a),
                          std::get<3>(a) - time_point,
                          std::get<4>(a),
                          std::move(args));
    }

    for (auto& k : kernel_events_v2) {
      auto& src_args = std::get<5>(k);
      decltype(EventRecord::args) args(src_args.begin(), src_args.end());
      events.emplace_back(EventCategory::KERNEL_EVENT,
                          std::get<1>(k),
                          std::get<2>(k),
                          std::get<0>(k),
                          std::get<3>(k) - time_point,
                          std::get<4>(k),
                          std::move(args));
    }
  } else {
    // Fall back to v1 API
    std::vector<EventInfo> api_events;
    std::vector<EventInfo> kernel_events;
    profiler_collect(api_events, kernel_events);

    decltype(EventRecord::args) event_args;

    for (auto& a : api_events) {
      events.emplace_back(EventCategory::API_EVENT,
                          std::get<1>(a),
                          std::get<2>(a),
                          std::get<0>(a),
                          std::get<3>(a) - time_point,
                          std::get<4>(a),
                          event_args);
    }

    for (auto& k : kernel_events) {
      events.emplace_back(EventCategory::KERNEL_EVENT,
                          std::get<1>(k),
                          std::get<2>(k),
                          std::get<0>(k),
                          std::get<3>(k) - time_point,
                          std::get<4>(k),
                          event_args);
    }
  }
}

#endif

}  // namespace profiling
}  // namespace onnxruntime
