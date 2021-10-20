// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/profiler_common.h"

#if !(defined(USE_ROCM) || defined(ENABLE_TRAINING))

#include "core/platform/ort_mutex.h"
#include <cupti.h>
#include <atomic>
#include <mutex>
#include <vector>

namespace onnxruntime {

namespace profiling {

using Events = std::vector<onnxruntime::profiling::EventRecord>;

class CudaProfiler final : public EpProfiler {
 public:
  CudaProfiler() = default;
  CudaProfiler(const CudaProfiler&) = delete;
  CudaProfiler& operator=(const CudaProfiler&) = delete;
  CudaProfiler(CudaProfiler&& cuda_profiler) noexcept {
    initialized_ = cuda_profiler.initialized_;
    cuda_profiler.initialized_ = false;
  }
  CudaProfiler& operator=(CudaProfiler&& cuda_profiler) noexcept {
    initialized_ = cuda_profiler.initialized_;
    cuda_profiler.initialized_ = false;
    return *this;
  }
  ~CudaProfiler();
  bool StartProfiling() override;
  void EndProfiling(TimePoint start_time, Events& events) override;
  void Start(uint64_t) override;
  void Stop(uint64_t) override;

 private:
  static void CUPTIAPI BufferRequested(uint8_t**, size_t*, size_t*);
  static void CUPTIAPI BufferCompleted(CUcontext, uint32_t, uint8_t*, size_t, size_t);
  struct KernelStat {
    std::string name_ = {};
    uint32_t stream_ = 0;
    int32_t grid_x_ = 0;
    int32_t grid_y_ = 0;
    int32_t grid_z_ = 0;
    int32_t block_x_ = 0;
    int32_t block_y_ = 0;
    int32_t block_z_ = 0;
    int64_t start_ = 0;
    int64_t stop_ = 0;
    uint32_t correlation_id = 0;
  };
  static std::atomic_flag enabled;
  static std::vector<KernelStat> stats;
  static std::unordered_map<uint32_t, uint64_t> id_map;

  void DisableEvents();
  void Clear();
  bool initialized_ = false;
};

}  // namespace profiling
}  // namespace onnxruntime

#else

namespace onnxruntime {

namespace profiling {

class CudaProfiler final : public EpProfiler {
 public:
  bool StartProfiling() override { return true; }
  void EndProfiling(TimePoint, Events&) override{};
  void Start(uint64_t) override{};
  void Stop(uint64_t) override{};
};

}
}

#endif
