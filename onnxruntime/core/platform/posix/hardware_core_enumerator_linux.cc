// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hardware_core_enumerator_linux.h"

#ifdef __linux__
#include <cpuid.h>
#include <vector>

namespace onnxruntime {

CheckIntelResult CheckIntel() {
  CheckIntelResult intel_check = {false, false};
  bool is_intel_specified_platform = false;
  unsigned int regs[4] = {0};
  __get_cpuid(0, &regs[0], &regs[1], &regs[2], &regs[3]);

  const unsigned int kVendorID_Intel[] = {0x756e6547, 0x6c65746e, 0x49656e69};  // "GenuntelineI"
  bool is_intel = (regs[1] == kVendorID_Intel[0] &&
                  regs[2] == kVendorID_Intel[1] &&
                  regs[3] == kVendorID_Intel[2]);
  if (!is_intel) {
    return intel_check;  // if not an Intel CPU, return early
  }

  __get_cpuid(1, &regs[0], &regs[1], &regs[2], &regs[3]);

  unsigned int base_family = (regs[0] >> 8) & 0xF;
  unsigned int base_model = (regs[0] >> 4) & 0xF;
  unsigned int extended_model = (regs[0] >> 16) & 0xF;

  unsigned int model =
      (base_family == 0x6 || base_family == 0xF)
          ? (base_model + (extended_model << 4))
          : base_model;

  constexpr unsigned int kVendorID_IntelSpecifiedPlatformIDs[] = {
      // ExtendedModel, ExtendedFamily, Family Code, and Model Number
      170,  // MTL (0xAA)
      197,  // ARL-H (0xC5)
      198   // ARL-U (0xC6)
  };

  for (auto id : kVendorID_IntelSpecifiedPlatformIDs) {
    if (model == id) {
      is_intel_specified_platform = true;
      break;
    }
  }

  intel_check.is_intel = is_intel;
  intel_check.is_intel_specified_platform = is_intel_specified_platform;

  return intel_check;
}
}  // namespace onnxruntime
#endif  //__linux__
