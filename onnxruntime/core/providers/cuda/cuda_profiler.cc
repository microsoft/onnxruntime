// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING)

#include <map>
#include <string>
#include <iostream>

#include "cupti_manager.h"
#include "cuda_profiler.h"


namespace onnxruntime {
namespace profiling {

// audupa: Debugging only, delete before merging
// #define CUDA_VERSION 11600

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000

CudaProfiler::CudaProfiler() {
  auto& manager = CUPTIManager::GetInstance();
  client_handle_ = manager.RegisterClient();
}

CudaProfiler::~CudaProfiler() {
  auto& manager = CUPTIManager::GetInstance();
  manager.DeregisterClient(client_handle_);
}

bool CudaProfiler::StartProfiling(TimePoint profiling_start_time) {
  auto& manager = CUPTIManager::GetInstance();
  manager.StartLogging();
  profiling_start_time_ = profiling_start_time;
  return true;
}

void CudaProfiler::EndProfiling(TimePoint start_time, Events& events) {
  auto& manager = CUPTIManager::GetInstance();
  std::map<uint64_t, Events> event_map;
  manager.Consume(client_handle_, start_time, event_map);
  MergeEvents(event_map, events);
}

void CudaProfiler::Start(uint64_t id) {
  auto& manager = CUPTIManager::GetInstance();
  manager.PushCorrelation(client_handle_, id, profiling_start_time_);
}

void CudaProfiler::Stop(uint64_t) {
  auto& manager = CUPTIManager::GetInstance();
  manager.PopCorrelation();
}

#else  // for cuda 10.x, no profiling

bool CudaProfiler::StartProfiling(TimePoint) { return false; }
void CudaProfiler::EndProfiling(TimePoint, Events&) {}
CudaProfiler::~CudaProfiler() {}
void CudaProfiler::Start(uint64_t) {}
void CudaProfiler::Stop(uint64_t) {}

#endif

}  // namespace profiling
}  // namespace onnxruntime
#endif
