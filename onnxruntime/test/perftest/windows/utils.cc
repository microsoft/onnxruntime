// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/perftest/utils.h"

#include <cstdint>

#include <Windows.h>
#include <psapi.h>


namespace onnxruntime {
namespace perftest {
namespace utils {

size_t GetPeakWorkingSetSize() {
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return pmc.PeakWorkingSetSize;
  }

  return 0;
}

static std::uint64_t SubtractFILETIME(const FILETIME& ft_a, const FILETIME& ft_b) {
  LARGE_INTEGER a, b;
  a.LowPart = ft_a.dwLowDateTime;
  a.HighPart = ft_a.dwHighDateTime;

  b.LowPart = ft_b.dwLowDateTime;
  b.HighPart = ft_b.dwHighDateTime;

  return a.QuadPart - b.QuadPart;
}

class CPUUsage : public ICPUUsage {
 public:
  CPUUsage() {
    Reset();
  }

  short GetUsage() const override {
    FILETIME sys_idle_ft, sys_kernel_ft, sys_user_ft, proc_creation_ft, proc_exit_ft, proc_kernel_ft, proc_user_ft;
    GetSystemTimes(&sys_idle_ft, &sys_kernel_ft, &sys_user_ft);
    GetProcessTimes(GetCurrentProcess(), &proc_creation_ft, &proc_exit_ft, &proc_kernel_ft, &proc_user_ft);

    std::uint64_t sys_kernel_ft_diff = SubtractFILETIME(sys_kernel_ft, sys_kernel_ft_);
    std::uint64_t sys_user_ft_diff = SubtractFILETIME(sys_user_ft, sys_user_ft_);

    std::uint64_t proc_kernel_diff = SubtractFILETIME(proc_kernel_ft, proc_kernel_ft_);
    std::uint64_t proc_user_diff = SubtractFILETIME(proc_user_ft, proc_user_ft_);

    std::uint64_t total_sys = sys_kernel_ft_diff + sys_user_ft_diff;
    std::uint64_t total_proc = proc_kernel_diff + proc_user_diff;

    return total_sys > 0 ? static_cast<short>((100.0 * total_proc) / total_sys) : 0;
  }

  void Reset() override {
    FILETIME sys_idle_ft, proc_creation_ft, proc_exit_ft;
    GetSystemTimes(&sys_idle_ft, &sys_kernel_ft_, &sys_user_ft_);
    GetProcessTimes(GetCurrentProcess(), &proc_creation_ft, &proc_exit_ft, &proc_kernel_ft_, &proc_user_ft_);
  }

 private:
  //system total times
  FILETIME sys_kernel_ft_;
  FILETIME sys_user_ft_;

  //process times
  FILETIME proc_kernel_ft_;
  FILETIME proc_user_ft_;
};

std::unique_ptr<ICPUUsage> CreateICPUUsage() {
  return std::make_unique<CPUUsage>();
}

}  // namespace utils
}  // namespace perftest
}  // namespace onnxruntime
