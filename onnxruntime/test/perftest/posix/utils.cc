// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/perftest/utils.h"

#include <cstddef>

#include <sys/times.h>
#include <sys/resource.h>

#include "core/platform/env.h"

namespace onnxruntime {
namespace perftest {
namespace utils {

std::size_t GetPeakWorkingSetSize() {
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return static_cast<size_t>(rusage.ru_maxrss * 1024L);
}

class CPUUsage : public ICPUUsage {
 public:
  CPUUsage() {
    Reset();
  }

  short GetUsage() const override {
    struct tms time_sample;
    clock_t total_clock_now = times(&time_sample);
    if (total_clock_now <= total_clock_start_ ||
        time_sample.tms_stime < proc_sys_clock_start_ ||
        time_sample.tms_utime < proc_user_clock_start_) {
      // overflow detection
      return -1;
    } else {
      clock_t proc_total_clock_diff = (time_sample.tms_stime - proc_sys_clock_start_) + (time_sample.tms_utime - proc_user_clock_start_);
      clock_t total_clock_diff = total_clock_now - total_clock_start_;
      return static_cast<short>(100.0 * proc_total_clock_diff / total_clock_diff / onnxruntime::Env::Default().GetNumCpuCores());
    }
  }

  void Reset() override {
    struct tms time_sample;
    total_clock_start_ = times(&time_sample);
    proc_sys_clock_start_ = time_sample.tms_stime;
    proc_user_clock_start_ = time_sample.tms_utime;
  }

 private:
  clock_t total_clock_start_;
  clock_t proc_sys_clock_start_;
  clock_t proc_user_clock_start_;
};

std::unique_ptr<ICPUUsage> CreateICPUUsage() {
  return std::make_unique<CPUUsage>();
}

}  // namespace utils
}  // namespace perftest
}  // namespace onnxruntime
