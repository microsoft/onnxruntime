// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <vector>
#include <map>

#include "core/common/gpu_profiler_common.h"
#include "roctracer_manager.h"

#if defined(USE_ROCM) && defined(ENABLE_ROCM_PROFILING)

namespace onnxruntime {
namespace profiling {

using Events = std::vector<onnxruntime::profiling::EventRecord>;

class RocmProfiler final : public GPUProfilerBase<RoctracerManager> {
 public:
  RocmProfiler();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RocmProfiler);
  ~RocmProfiler();
};

}  // namespace profiling
}  // namespace onnxruntime

#else

namespace onnxruntime {
namespace profiling {

class RocmProfiler final : public EpProfiler {
 public:
  RocmProfiler() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RocmProfiler);
  ~RocmProfiler() {}
  bool StartProfiling(TimePoint) override { return true; }
  void EndProfiling(TimePoint, Events&) override{};
  void Start(uint64_t) override{};
  void Stop(uint64_t) override{};
};

}  // namespace profiling
}  // namespace onnxruntime

#endif
