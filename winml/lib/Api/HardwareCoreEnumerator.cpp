
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api/pch/pch.h"
#include "HardwareCoreEnumerator.h"

namespace WINMLP {

struct LogicalProcessorInformation {
  std::unique_ptr<char[]> Buffer;
  size_t Length;
};

struct CoreCounter {
  uint32_t PhysicalCores = 0;
  uint32_t LLCCores = 0;
};

static LogicalProcessorInformation GetLogicalProcessorInfos(LOGICAL_PROCESSOR_RELATIONSHIP relationship) {
  DWORD length = 0;
  DWORD rc = GetLogicalProcessorInformationEx(relationship, nullptr, &length);

  assert(rc == FALSE);

  auto processorInformationBytes = std::make_unique<char[]>(length);

  rc = GetLogicalProcessorInformationEx(
    relationship, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(processorInformationBytes.get()), &length
  );

  assert(rc == TRUE);

  return {std::move(processorInformationBytes), length};
}

uint32_t CountSetBits(DWORD input) {
  uint32_t c;
  for (c = 0; input; c++) {
    input &= input - 1;
  }
  return c;
}

static CoreCounter GetCoreInfo() {
  auto logicalProcessorInformation = GetLogicalProcessorInfos(RelationAll);

  CoreCounter cores;
  DWORD dwLevel2GroupMask = 0;
  DWORD dwLevel3GroupMask = 0;
  size_t read = 0;
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX currentProcessorInfo = NULL;

  while ((read + FIELD_OFFSET(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor)) < logicalProcessorInformation.Length
  ) {
    currentProcessorInfo =
      reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(logicalProcessorInformation.Buffer.get() + read);
    if ((read + currentProcessorInfo->Size) > logicalProcessorInformation.Length) {
      break;
    }

    switch (currentProcessorInfo->Relationship) {
      case RelationProcessorCore:
        cores.PhysicalCores++;
        break;
      case RelationCache:
        //Cache level masks count Logicial processors
        if (currentProcessorInfo->Cache.Level == 2) {
          dwLevel2GroupMask |= currentProcessorInfo->Cache.GroupMask.Mask;
        } else if (currentProcessorInfo->Cache.Level == 3) {
          dwLevel3GroupMask |= currentProcessorInfo->Cache.GroupMask.Mask;
        }
        break;
    }

    read += currentProcessorInfo->Size;
  }

  cores.LLCCores = cores.PhysicalCores - CountSetBits(dwLevel2GroupMask & ~dwLevel3GroupMask);

  return cores;
}

uint32_t HardwareCoreEnumerator::DefaultIntraOpNumThreads() {
  // # of physical cores = # of P cores + # of E Cores + # of Soc Cores.
  // # of logical cores = # of P cores x 2 (if hyper threading is enabled) + # of E cores + # of Soc Cores.
  auto cores = GetCoreInfo();

#if !defined(_M_ARM64EC) && !defined(_M_ARM64) && !defined(__aarch64__)
  const int kVendorID_Intel[3] = {0x756e6547, 0x6c65746e, 0x49656e69};  // "GenuntelineI"
  bool isIntelSpecifiedPlatform = false;
  const int kVendorID_IntelSpecifiedPlatformIDs[3] = {
    // ExtendedModel,ExtendedFamily,Family Code, and Model Number
    0xa06a,  // MTL
    0xc065,  // ARL-H
    0xb065   // ARL-U
  };

  int regs_leaf0[4];
  int regs_leaf1[4];
  __cpuid(regs_leaf0, 0);
  __cpuid(regs_leaf1, 0x1);

  auto isIntel = (kVendorID_Intel[0] == regs_leaf0[1]) && (kVendorID_Intel[1] == regs_leaf0[2]) &&
    (kVendorID_Intel[2] == regs_leaf0[3]);

  for (int intelSpecifiedPlatform : kVendorID_IntelSpecifiedPlatformIDs) {
    if ((regs_leaf1[0] >> 4) == intelSpecifiedPlatform) {
      isIntelSpecifiedPlatform = true;
    }
  }

  if (isIntel && isIntelSpecifiedPlatform) {
    // We want to use the number of physical cores, but exclude cores without an LLC
    return cores.LLCCores;
  }
#endif
  return cores.PhysicalCores;
}

}  // namespace WINMLP
