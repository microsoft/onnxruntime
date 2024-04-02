﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hardware_core_enumerator.h"
#include <memory>
#include <Windows.h>
#include <assert.h>

namespace onnxruntime {

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
      relationship, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(processorInformationBytes.get()), &length);

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

  while ((read + FIELD_OFFSET(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor)) < logicalProcessorInformation.Length) {
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
        if (currentProcessorInfo->Cache.Level == 2) {
          dwLevel2GroupMask |= currentProcessorInfo->Cache.GroupMask.Mask;
        } else if (currentProcessorInfo->Cache.Level == 3) {
          dwLevel3GroupMask |= currentProcessorInfo->Cache.GroupMask.Mask;
        }
        break;
    }

    read += currentProcessorInfo->Size;
  }
  // Cores with L2 and LLC cache levels = # Physical Cores - # logical cores without LLC
  cores.LLCCores = cores.PhysicalCores - CountSetBits(dwLevel2GroupMask & ~dwLevel3GroupMask);

  return cores;
}

uint32_t HardwareCoreEnumerator::DefaultIntraOpNumThreads() {
  // # of physical cores = # of P cores + # of E Cores + # of Soc Cores.
  // # of logical cores = # of P cores x 2 (if hyper threading is enabled) + # of E cores + # of Soc Cores.
  auto cores = GetCoreInfo();

  return cores.LLCCores;
}

}  // namespace onnxruntime
