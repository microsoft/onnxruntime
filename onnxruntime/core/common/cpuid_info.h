// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

#if defined(_M_IX86) || (defined(_M_X64) && !defined(_M_ARM64EC)) || defined(__i386__) || defined(__x86_64__)
#define CPUIDINFO_ARCH_X86
#endif

#if defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
#define CPUIDINFO_ARCH_ARM
#endif

namespace onnxruntime {

class CPUIDInfo {
 public:
  static const CPUIDInfo& GetCPUIDInfo() {
    static CPUIDInfo cpuid_info;
    return cpuid_info;
  }

  bool HasAVX() const { return has_avx_; }
  bool HasAVX2() const { return has_avx2_; }
  bool HasAVX512f() const { return has_avx512f_; }
  bool HasAVX512Skylake() const { return has_avx512_skylake_; }
  bool HasF16C() const { return has_f16c_; }
  bool HasSSE3() const { return has_sse3_; }
  bool HasSSE4_1() const { return has_sse4_1_; }
  bool IsHybrid() const { return is_hybrid_; }

  // ARM 
  bool HasArmNeonDot() const { return has_arm_neon_dot_; }

  /**
   * @return CPU core micro-architecture running the current thread
  */
  int32_t GetCurrentUarch() const;

 private:
  CPUIDInfo();
  bool has_avx_{false};
  bool has_avx2_{false};
  bool has_avx512f_{false};
  bool has_avx512_skylake_{false};
  bool has_f16c_{false};
  bool has_sse3_{false};
  bool has_sse4_1_{false};
  bool is_hybrid_{false};

#if (defined(CPUIDINFO_ARCH_X86) || defined(CPUIDINFO_ARCH_ARM)) && defined(CPUINFO_SUPPORTED)
  bool pytorch_cpuinfo_init_{false};
#endif
  bool has_arm_neon_dot_{false};
};

}  // namespace onnxruntime
