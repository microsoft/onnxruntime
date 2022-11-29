// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000

#include <atomic>
#include <mutex>
#include <vector>

#include "core/common/gpu_profiler_common.h"
#include "cupti_manager.h"

namespace onnxruntime {
namespace profiling {

class CudaProfiler final : public GPUProfilerBase<CUPTIManager> {
 public:
  CudaProfiler();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaProfiler);
  ~CudaProfiler();
};

}  // namespace profiling
}  // namespace onnxruntime

#else /* !defined(USE_CUDA) || !defined(ENABLE_CUDA_PROFILING) || !defined(CUDA_VERSION) || CUDA_VERSION < 11000 */
namespace onnxruntime {
namespace profiling {

class CudaProfiler final : public EpProfiler {
 public:
  bool StartProfiling(TimePoint) override { return true; }
  void EndProfiling(TimePoint, Events&) override{};
  void Start(uint64_t) override{};
  void Stop(uint64_t) override{};
};

}  // namespace profiling
}  // namespace onnxruntime

#endif
