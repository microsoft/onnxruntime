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

bool RocmProfiler::StartProfiling(TimePoint profiling_start_time) {
  auto& manager = RoctracerManager::GetInstance();
  manager.StartLogging();
  profiling_start_time_ = profiling_start_time;
  return true;
}

void RocmProfiler::EndProfiling(TimePoint start_time, Events& events) {
  auto& manager = RoctracerManager::GetInstance();
  std::map<uint64_t, Events> event_map;
  manager.Consume(client_handle_, start_time, event_map);
  MergeEvents(event_map, events);
}

void RocmProfiler::Start(uint64_t id) {
  auto& manager = RoctracerManager::GetInstance();
  manager.PushCorrelation(client_handle_, id, profiling_start_time_);
}

void RocmProfiler::Stop(uint64_t id) {
  auto& manager = RoctracerManager::GetInstance();
  uint64_t unused;
  manager.PopCorrelation(unused);
}

}  // namespace profiling
}  // namespace onnxruntime
#endif
