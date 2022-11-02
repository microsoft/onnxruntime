// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <vector>
#include <map>

#include "core/common/profiler_common.h"

#if defined(USE_ROCM) && defined(ENABLE_ROCM_PROFILING)

namespace onnxruntime {
namespace profiling {

using Events = std::vector<onnxruntime::profiling::EventRecord>;

class RocmProfiler final : public EpProfiler {
 public:
  RocmProfiler();
  RocmProfiler(const RocmProfiler&) = delete;
  RocmProfiler& operator=(const RocmProfiler&) = delete;
  ~RocmProfiler();
  bool StartProfiling(TimePoint profiling_start_time) override;
  void EndProfiling(TimePoint start_time, Events& events) override;
  void Start(uint64_t) override;
  void Stop(uint64_t) override;

private:
  uint64_t client_handle_;
  TimePoint profiling_start_time_;
};

}  // namespace profiling
}  // namespace onnxruntime

#else

namespace onnxruntime {

namespace profiling {

class RocmProfiler final : public EpProfiler {
 public:
  bool StartProfiling(TimePoint) override { return true; }
  void EndProfiling(TimePoint, Events&) override{};
  void Start(uint64_t) override{};
  void Stop(uint64_t) override{};
};

}
}

#endif
