// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

class CPUIDInfo {
public:
  static const CPUIDInfo& GetCPUIDInfo() {
    static CPUIDInfo cpuid_info;
    return cpuid_info;
  }

  bool HasAVX2() const { return has_avx2_; }
  bool HasAVX512f() const { return has_avx512f_; }
  bool HasF16C() const { return has_f16c_; }

private:
  CPUIDInfo() noexcept;
  bool has_avx2_{false};
  bool has_avx512f_{false};
  bool has_f16c_{false};
};

}  // namespace onnxruntime
