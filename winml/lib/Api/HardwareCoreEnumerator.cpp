// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api/pch/pch.h"

#include "HardwareCoreEnumerator.h"

namespace WINMLP {

struct LogicalProcessorInformation {
  std::unique_ptr<char[]> Buffer;
  size_t Length;
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

static long long GetNumberOfSoCDieAtoms() {
  // while (Size > (ULONG)FIELD_OFFSET(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor)) {
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

static long long GetNumberOfCores() {
  auto logicalProcessorInformation = GetLogicalProcessorInfos(RelationProcessorCore);
  auto processorInformation = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)logicalProcessorInformation.Buffer.get();

  KAFFINITY coreMask = 0;
  size_t read = 0;
  while (read <= logicalProcessorInformation.Length) {
    switch (processorInformation->Relationship) {
      case RelationProcessorCore:
        coreMask |= processorInformation->Processor.GroupMask->Mask;
        break;
    }

    read += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX);
    processorInformation++;
  }
  return __popcnt64(coreMask);
}

uint32_t HardwareCoreEnumerator::DefaultIntraOpNumThreads() {
  auto get_number_of_cores = static_cast<uint32_t>(GetNumberOfCores());
  auto get_number_of_soc_die_atoms = static_cast<uint32_t>(GetNumberOfSoCDieAtoms());
  auto num_p_and_e_cores = get_number_of_cores - get_number_of_soc_die_atoms;
  printf("num_cores = %d, get_number_of_cores = %d, get_number_of_soc_die_atoms = %d\n", num_cores, get_number_of_cores,
get_number_of_soc_die_atoms);
  return num_p_and_e_cores;
}

}  // namespace WINMLP
