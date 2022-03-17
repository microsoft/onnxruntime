// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <vector>
#include <map>

#include "core/common/profiler_common.h"

#if defined(USE_ROCM) && defined(ENABLE_ROCM_PROFILING)

class RoctracerLogger;
class roctracerRow;

namespace onnxruntime {

namespace profiling {

using Events = std::vector<onnxruntime::profiling::EventRecord>;

class RocmProfiler final : public EpProfiler {
 public:
  RocmProfiler();
  RocmProfiler(const RocmProfiler&) = delete;
  RocmProfiler& operator=(const RocmProfiler&) = delete;
#if 0
  RocmProfiler(RocmProfiler&& rocm_profiler) noexcept {
    initialized_ = rocm_profiler.initialized_;
    rocm_profiler.initialized_ = false;
  }
  RocmProfiler& operator=(RocmProfiler&& rocm_profiler) noexcept {
    initialized_ = rocm_profiler.initialized_;
    rocm_profiler.initialized_ = false;
    return *this;
  }
#endif
  ~RocmProfiler();
  bool StartProfiling() override;
  void EndProfiling(TimePoint start_time, Events& events) override;
  void Start(uint64_t) override;
  void Stop(uint64_t) override;

 private:
  void addEventRecord(const roctracerRow &item, int64_t pstart, const std::initializer_list<std::pair<std::string, std::string>> &args, std::map<uint64_t, std::vector<EventRecord>> &event_map);

  RoctracerLogger *d;
};

}  // namespace profiling
}  // namespace onnxruntime

#else

namespace onnxruntime {

namespace profiling {

class RocmProfiler final : public EpProfiler {
 public:
  bool StartProfiling() override { return true; }
  void EndProfiling(TimePoint, Events&) override{};
  void Start(uint64_t) override{};
  void Stop(uint64_t) override{};
};

}
}




#endif
