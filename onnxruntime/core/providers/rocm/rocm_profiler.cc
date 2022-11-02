// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_ROCM) && defined(ENABLE_ROCM_PROFILING)

#include <chrono>
#include <time.h>

#include "core/common/profiler_common.h"
#include "core/providers/rocm/rocm_profiler.h"
#include "core/providers/rocm/roctracer_manager.h"

namespace onnxruntime {
namespace profiling {

RocmProfiler::RocmProfiler() {
  auto& manager = RoctracerManager::GetInstance();
  client_handle_ = manager.RegisterClient();
}

RocmProfiler::~RocmProfiler() {
  auto& manager = RoctracerManager::GetInstance();
  manager.DeregisterClient(client_handle_);
}

bool RocmProfiler::StartProfiling() {
  auto& manager = RoctracerManager::GetInstance();
  manager.StartLogging();
  return true;
}

void RocmProfiler::EndProfiling(TimePoint start_time, Events& events) {
  auto& manager = RoctracerManager::GetInstance();
  std::map<uint64_t, Events> event_map;
  manager.Consume(client_handle_, start_time, event_map);

  Events merged_events;

  auto event_iter = std::make_move_iterator(events.begin());
  auto event_end = std::make_move_iterator(events.end());
  for (auto& map_iter : event_map) {
    auto ts = static_cast<long long>(map_iter.first);
    while (event_iter != event_end && event_iter->ts < ts) {
      merged_events.emplace_back(*event_iter);
      ++event_iter;
    }

    // find the last event with the same timestamp.
    while (event_iter != event_end && event_iter->ts == ts && (event_iter + 1)->ts == ts) {
      ++event_iter;
    }

    if (event_iter != event_end && event_iter->ts == ts) {
      uint64_t increment = 1;
      for (auto& evt : map_iter.second) {
        evt.args["op_name"] = event_iter->args["op_name"];

        // roctracer timestamps don't use Jan 1 1970 as an epoch,
        // not sure what epoch it uses, but we adjust the timestamp
        // here to something sensible.
        evt.ts = event_iter->ts + increment;
        ++increment;
      }
      merged_events.emplace_back(*event_iter);
      ++event_iter;
    }

    merged_events.insert(merged_events.end(),
                         std::make_move_iterator(map_iter.second.begin()),
                         std::make_move_iterator(map_iter.second.end()));
  }

  // move any remaining events
  merged_events.insert(merged_events.end(), event_iter, event_end);
  std::swap(events, merged_events);
}

void RocmProfiler::Start(uint64_t id) {
  auto& manager = RoctracerManager::GetInstance();
  manager.PushCorrelation(client_handle_, id);
}

void RocmProfiler::Stop(uint64_t id) {
  auto& manager = RoctracerManager::GetInstance();
  uint64_t unused;
  manager.PopCorrelation(unused);
}

}  // namespace profiling
}  // namespace onnxruntime
#endif
