// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cpuinfo.h>

#include <Windows.h>

#include <cstdlib>
#include <iostream>

#include "core/common/cpuid_info.h"

namespace {
const void* allocation = nullptr;
bool* allocation_freed = nullptr;

void RecordCpuinfoDeinitialization() {
  *allocation_freed = allocation != nullptr && !HeapValidate(GetProcessHeap(), 0, allocation);
}
}  // namespace

extern "C" const void* OrtGetCpuinfoAllocationForTesting(bool* freed) {
  if (freed == nullptr) {
    std::cerr << "Missing cpuinfo deinitialization result" << std::endl;
    return nullptr;
  }

  allocation_freed = freed;
  *allocation_freed = false;
  // Register first so this runs after CPUIDInfo destruction, before remaining CRT/loader cleanup
  // can reuse the freed address and make a later HeapValidate check observe a different allocation.
  if (std::atexit(RecordCpuinfoDeinitialization) != 0) {
    std::cerr << "Could not register cpuinfo deinitialization observer" << std::endl;
    return nullptr;
  }

  static_cast<void>(onnxruntime::CPUIDInfo::GetCPUIDInfo());
  allocation = cpuinfo_get_processors();
  return allocation;
}
