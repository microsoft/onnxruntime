// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {

class CPUIDInfo {
 public:
  static const CPUIDInfo& GetCPUIDInfo() {
    static CPUIDInfo cpuid_info;
    return cpuid_info;
  }

  bool HasAVX() const noexcept { return has_avx_; }
  bool HasAVX2() const noexcept { return has_avx2_; }
  bool HasAVX512f() const noexcept { return has_avx512f_; }
  bool HasAVX512Skylake() const noexcept { return has_avx512_skylake_; }
  bool HasF16C() const noexcept { return has_f16c_; }
  bool HasSSE3() const noexcept { return has_sse3_; }
  bool HasSSE4_1() const noexcept { return has_sse4_1_; }
  bool IsHybrid() const noexcept { return is_hybrid_; }

  // ARM 
  bool HasArmNeonDot() const noexcept { return has_arm_neon_dot_; }

 private:
  GSL_SUPPRESS(f .6)
  GSL_SUPPRESS(bounds .4)
  CPUIDInfo();
  bool has_avx_{false};
  bool has_avx2_{false};
  bool has_avx512f_{false};
  bool has_avx512_skylake_{false};
  bool has_f16c_{false};
  bool has_sse3_{false};
  bool has_sse4_1_{false};
  bool is_hybrid_{false};

  bool has_arm_neon_dot_{false};
  //TODO: the constructor shouldn't throw
  static CPUIDInfo instance_;
};

}  // namespace onnxruntime
