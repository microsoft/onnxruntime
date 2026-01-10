// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/check_intel.h"

#if (defined(_M_AMD64) || defined(__x86_64__))
#if defined(__linux__)
#include <cpuid.h>
#elif defined(_WIN32)
#include <intrin.h>
#endif
#endif

namespace onnxruntime {

CheckIntelResult CheckIntel() {
  CheckIntelResult intel_check = {false, false};
  bool is_intel = false;
  bool is_intel_specified_platform = false;

#if (defined(_M_AMD64) || defined(__x86_64__))
#if defined(_WIN32)
  constexpr unsigned int kVendorID_Intel[] = {0x756e6547, 0x6c65746e, 0x49656e69};  // "GenuntelineI"
  constexpr unsigned int kVendorID_IntelSpecifiedPlatformIDs[] = {
      // ExtendedModel, ExtendedFamily, Family Code, and Model Number
      0xa06a,  // MTL
      0xc065,  // ARL-H
      0xb065   // ARL-U
  };

  int regs_leaf0[4];
  int regs_leaf1[4];
  __cpuid(regs_leaf0, 0);
  __cpuid(regs_leaf1, 0x1);

  is_intel =
      (kVendorID_Intel[0] == static_cast<unsigned int>(regs_leaf0[1])) &&
      (kVendorID_Intel[1] == static_cast<unsigned int>(regs_leaf0[2])) &&
      (kVendorID_Intel[2] == static_cast<unsigned int>(regs_leaf0[3]));

  if (!is_intel) {
    return intel_check;  // if not an Intel CPU, return early
  }

  for (auto intel_specified_platform : kVendorID_IntelSpecifiedPlatformIDs) {
    if ((static_cast<unsigned int>(regs_leaf1[0]) >> 4) == intel_specified_platform) {
      is_intel_specified_platform = true;
      break;
    }
  }

#elif defined(__linux__)
  constexpr unsigned int kVendorID_Intel[] = {0x756e6547, 0x6c65746e, 0x49656e69};  // "GenuntelineI"
  unsigned int regs[4] = {0};
  __get_cpuid(0, &regs[0], &regs[1], &regs[2], &regs[3]);

  is_intel = (regs[1] == kVendorID_Intel[0] &&
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
#endif  //__linux__
#endif  // (_M_AMD64) || (__x86_64__)

  intel_check.is_intel = is_intel;
  intel_check.is_intel_specified_platform = is_intel_specified_platform;

  return intel_check;
}

}  // namespace onnxruntime
