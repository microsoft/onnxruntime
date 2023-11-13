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
  long long PhysicalCores = 0;
  long long SocDieCores = 0;
};

static LogicalProcessorInformation GetLogicalProcessorInfos(LOGICAL_PROCESSOR_RELATIONSHIP relationship) {
  DWORD length = 0;
  DWORD rc = GetLogicalProcessorInformationEx(relationship, nullptr, &length);

  assert(rc == FALSE);

  auto processorInformationBytes = std::make_unique<char[]>(length);

  rc = GetLogicalProcessorInformationEx(
    relationship, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)processorInformationBytes.get(), &length
  );

  assert(rc == TRUE);

  return {std::move(processorInformationBytes), length};
}

static long long GetNumberOfSoCDieCores() {
  DWORD dwLevel2GroupMask = 0;
  DWORD dwLevel3GroupMask = 0;
  DWORD dwSoCGroupMask = 0;

  auto logicalProcessorInformation = GetLogicalProcessorInfos(RelationAll);
  auto processorInformation = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)logicalProcessorInformation.Buffer.get();

  size_t read = 0;
  while (read <= logicalProcessorInformation.Length) {
    switch (processorInformation->Relationship) {
      case RelationCache:
        if (processorInformation->Cache.Level == 2) {
          dwLevel2GroupMask |= processorInformation->Cache.GroupMask.Mask;
        } else if (processorInformation->Cache.Level == 3) {
          dwLevel3GroupMask |= processorInformation->Cache.GroupMask.Mask;
        }
        break;
    }

    read += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX);
    processorInformation++;
  }

  dwSoCGroupMask = (dwLevel2GroupMask & ~dwLevel3GroupMask);

  return __popcnt(dwSoCGroupMask);
}

static CoreCounter GetNumberOPhysicalAndEngineeringCores() {
  auto logicalProcessorInformation = GetLogicalProcessorInfos(RelationProcessorCore);

  CoreCounter cores;
  size_t read = 0;
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX currentProcessorInfo = NULL;

  while ((read + FIELD_OFFSET(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor)) < logicalProcessorInformation.Length) {
    currentProcessorInfo = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(logicalProcessorInformation.Buffer.get() + read);
    if ((read + currentProcessorInfo->Size) > logicalProcessorInformation.Length) {
      break;
    }

    switch (currentProcessorInfo->Relationship) {
      case RelationProcessorCore:
        cores.PhysicalCores++;
        break;
    }

    read += currentProcessorInfo->Size;
  }

  cores.SocDieCores = GetNumberOfSoCDieCores();
  return cores;
}

uint32_t HardwareCoreEnumerator::DefaultIntraOpNumThreads() {
  // # of physical cores = # of P cores + # of E Cores + # of Soc Cores.
  // # of logical cores = # of P cores x 2 (if hyper threading is enabled) + # of E cores + # of Soc Cores.
  auto cores = GetNumberOPhysicalAndEngineeringCores();
  return static_cast<uint32_t>(cores.PhysicalCores - cores.SocDieCores);
}

}  // namespace WINMLP
