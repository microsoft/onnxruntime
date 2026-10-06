// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cpuinfo.h>

#include "core/common/cpuid_info.h"

extern "C" const void* OrtGetCpuinfoAllocationForTesting() {
  static_cast<void>(onnxruntime::CPUIDInfo::GetCPUIDInfo());
  return cpuinfo_get_processors();
}
