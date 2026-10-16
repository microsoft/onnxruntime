// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cpuinfo.h>

#include "core/common/cpuid_info.h"

extern "C" {
extern bool cpuinfo_is_initialized;
extern struct cpuinfo_processor* cpuinfo_processors;
extern uint32_t cpuinfo_processors_count;
}

namespace {

bool IsProcessorStateReleased() {
  return !cpuinfo_is_initialized && cpuinfo_processors == nullptr && cpuinfo_processors_count == 0;
}

struct CpuinfoUnloadObserver {
  explicit CpuinfoUnloadObserver(int* state) : unload_state{state} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CpuinfoUnloadObserver);

  int* unload_state;
  bool retained_reference = false;

  ~CpuinfoUnloadObserver() {
    *unload_state = IsProcessorStateReleased() ? 0 : 1;
    if (retained_reference) {
      cpuinfo_deinitialize();
      if (!IsProcessorStateReleased()) {
        *unload_state = 2;
      }
    }
  }
};

}  // namespace

extern "C" const void* OrtGetCpuinfoAllocationForTesting(int* unload_state, bool retain_reference) {
  // Reverse destruction order checks cpuinfo after CPUIDInfo releases its reference, before the DLL is unmapped.
  static CpuinfoUnloadObserver observer{unload_state};
  static_cast<void>(onnxruntime::CPUIDInfo::GetCPUIDInfo());
  if (!cpuinfo_is_initialized) {
    return nullptr;
  }
  if (retain_reference) {
    observer.retained_reference = cpuinfo_initialize();
    if (!observer.retained_reference) {
      return nullptr;
    }
  }
  return cpuinfo_get_processors();
}
