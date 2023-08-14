// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING)

#include <atomic>
#include <mutex>
#include <vector>

#include <cupti.h>

// Do not move the check for CUDA_VERSION above #include <cupti.h>
// the macros are defined in cupti.h
#if defined(USE_CUDA)

#include "core/common/gpu_profiler_common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace profiling {

class CUPTIManager : public GPUTracerManager<CUPTIManager> {
  friend class GPUTracerManager<CUPTIManager>;

 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CUPTIManager);
  ~CUPTIManager();
  static CUPTIManager& GetInstance();

 protected:
  bool PushUniqueCorrelation(uint64_t unique_cid);
  void PopUniqueCorrelation(uint64_t& popped_unique_cid);
  bool OnStartLogging();
  void OnStopLogging();
  void ProcessActivityBuffers(const std::vector<ProfilerActivityBuffer>& buffers,
                              const TimePoint& start_time);
  void FlushActivities();
  uint64_t GetGPUTimestampInNanoseconds();

 private:
  static constexpr size_t kActivityBufferSize = 32 * 1024;

  CUPTIManager() = default;

  static void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
  static void CUPTIAPI BufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid_size);
}; /* class CUPTIManager*/

} /* namespace profiling */
} /* namespace onnxruntime */

#endif /* #if defined(USE_CUDA) */
#endif /* #if defined (USE_CUDA) && defined(ENABLE_CUDA_PROFILING) */
