// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/cpuid_arch_definition.h"

namespace onnxruntime {

class CPUIDInfo {
 public:
  static const CPUIDInfo& GetCPUIDInfo() {
    static CPUIDInfo cpuid_info;
    return cpuid_info;
  }

  bool HasAMX_BF16() const { return has_amx_bf16_; }
  bool HasAVX() const { return has_avx_; }
  bool HasAVX2() const { return has_avx2_; }
  bool HasAVX512f() const { return has_avx512f_; }
  bool HasAVX512_BF16() const { return has_avx512_bf16_; }
  bool HasAVX512Skylake() const { return has_avx512_skylake_; }
  bool HasF16C() const { return has_f16c_; } /*fp16 conversion inst*/
  bool HasSSE3() const { return has_sse3_; }
  bool HasSSE4_1() const { return has_sse4_1_; }
  bool IsHybrid() const { return is_hybrid_; }

  // ARM
  bool HasArmNeonDot() const { return has_arm_neon_dot_; }
  bool HasArmNeon_I8MM() const { return has_arm_neon_i8mm_; }
  bool HasArmSVE_I8MM() const { return has_arm_sve_i8mm_; }
  bool HasArmNeon_BF16() const { return has_arm_neon_bf16_; }

  uint32_t GetCurrentCoreIdx() const;

  /**
   * @return CPU core micro-architecture running the current thread
   */
  int32_t GetCurrentUarch() const {
    if (core_uarchs_.empty()) {
      return -1;
    }

    uint32_t coreIdx = GetCurrentCoreIdx();
    if (coreIdx >= core_uarchs_.size()) {
      return -1;
    }
    return core_uarchs_[coreIdx];
  }

  /**
   * @return CPU core micro-architecture
   */
  int32_t GetCoreUarch(uint32_t coreId) const {
    if (coreId >= core_uarchs_.size()) {
      return -1;
    }
    return core_uarchs_[coreId];
  }

  /**
   * @brief Some ARMv8 power efficient core has narrower 64b load/store
   *        that needs specialized optimiztion in kernels
   * @return whether the indicated core has narrower load/store device
   */
  bool IsCoreArmv8NarrowLd(uint32_t coreId) const {
    if (coreId >= is_armv8_narrow_ld_.size()) {
      return false;
    }
    return is_armv8_narrow_ld_[coreId];
  }

  /**
   * @brief Some ARMv8 power efficient core has narrower 64b load/store
   *        that needs specialized optimiztion in kernels
   * @return whether the current core has narrower load/store device
   */
  bool IsCurrentCoreArmv8NarrowLd() const {
    if (is_armv8_narrow_ld_.empty()) {
      return false;
    }

    uint32_t coreIdx = GetCurrentCoreIdx();
    if (coreIdx >= is_armv8_narrow_ld_.size()) {
      return false;
    }
    return is_armv8_narrow_ld_[coreIdx];
  }

  bool HasFp16VectorAcceleration() const {
    return has_fp16_;
  }

 private:
  CPUIDInfo();
  bool has_amx_bf16_{false};
  bool has_avx_{false};
  bool has_avx2_{false};
  bool has_avx512f_{false};
  bool has_avx512_bf16_{false};
  bool has_avx512_skylake_{false};
  bool has_f16c_{false};
  bool has_sse3_{false};
  bool has_sse4_1_{false};
  bool is_hybrid_{false};

  std::vector<uint32_t> core_uarchs_;  // micro-arch of each core

  // In ARMv8 systems, some power efficient cores has narrower
  // 64b load/store devices. It takes longer for them to load
  // 128b vectore registers.
  std::vector<bool> is_armv8_narrow_ld_;

  bool has_arm_neon_dot_{false};
  bool has_fp16_{false};
  bool has_arm_neon_i8mm_{false};
  bool has_arm_sve_i8mm_{false};
  bool has_arm_neon_bf16_{false};

#ifdef CPUIDINFO_ARCH_X86

  void X86Init();
#elif defined(CPUIDINFO_ARCH_ARM)
  // Now the following var is only used in ARM build, but later one we may expand the usage.
  bool pytorch_cpuinfo_init_{false};
#endif

#ifdef __linux__

  void ArmLinuxInit();

#elif defined(_WIN32)

  void ArmWindowsInit();

#endif /* (arm or arm64) and windows */
};

}  // namespace onnxruntime
