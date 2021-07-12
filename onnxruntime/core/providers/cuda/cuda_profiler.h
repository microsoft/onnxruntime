// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/platform/ort_mutex.h"
#include "core/common/logging/logging.h"
#include <cupti.h>
#include <mutex>
#include <vector>

namespace onnxruntime {

namespace cuda {

using TimePoint = std::chrono::high_resolution_clock::time_point;
using Events = std::vector<onnxruntime::profiling::EventRecord>;

class CudaProfiler final {
 public:
  CudaProfiler() = delete;
  CudaProfiler(const CudaProfiler&) = delete;
  CudaProfiler& operator=(const CudaProfiler&) = delete;
  CudaProfiler(CudaProfiler&&) = delete;
  CudaProfiler& operator=(CudaProfiler&&) = delete;
  ~CudaProfiler() = default;

  static void StartProfiling(TimePoint, int, int);
  static Events StopProfiling();

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
    int64_t correlation_id = 0;
  };
  static onnxruntime::OrtMutex mtx;
  static std::atomic_flag enabled;
  static std::vector<KernelStat> stats;
  static bool initialized;
  static TimePoint start_time;
  static int pid;
  static int tid;
};

}  // namespace cuda
}  // namespace onnxruntime