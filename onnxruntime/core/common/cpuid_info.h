// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {

class CPUIDInfo {
 public:
  static common::Status Initialize() {
    return instance_.Init();
  }
  static const CPUIDInfo& GetCPUIDInfo() {
    if (!instance_.initalized_) {
      ORT_THROW("CPUIDInfo used before initialization!");
    }
    return instance_;
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

 private:
  common::Status Init();
  bool initalized_{false};
  bool has_avx_{false};
  bool has_avx2_{false};
  bool has_avx512f_{false};
  bool has_avx512_skylake_{false};
  bool has_f16c_{false};
  bool has_sse3_{false};
  bool has_sse4_1_{false};
  bool is_hybrid_{false};

  bool has_arm_neon_dot_{false};

  static CPUIDInfo instance_;
};

}  // namespace onnxruntime
