// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING)
#include <atomic>
#include <mutex>
#include <vector>

#include "core/common/gpu_profiler_common.h"

namespace onnxruntime {
namespace profiling {

using Events = std::vector<onnxruntime::profiling::EventRecord>;

class CudaProfiler final : public GPUProfilerBase {
 public:
  CudaProfiler();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaProfiler);
  ~CudaProfiler();
  bool StartProfiling(TimePoint profiling_start_time) override;
  void EndProfiling(TimePoint start_time, Events& events) override;
  void Start(uint64_t) override;
  void Stop(uint64_t) override;

private:
  uint64_t client_handle_ = 0;
  TimePoint profiling_start_time_{};
};

}  // namespace profiling
}  // namespace onnxruntime

#else

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
