// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cpuinfo.h>

#include "core/common/cpuid_info.h"

extern "C" {
extern bool cpuinfo_is_initialized;
extern struct cpuinfo_processor* cpuinfo_processors;
}

namespace {
struct CpuinfoTeardownObserver {
  bool* deinitialized;

  ~CpuinfoTeardownObserver() {
    // A freed heap address may already be reused by the time FreeLibrary returns.
    *deinitialized = !cpuinfo_is_initialized && cpuinfo_processors == nullptr;
  }
};
}  // namespace

extern "C" bool OrtInitializeCpuinfoForTesting(bool* deinitialized) {
  // Construct first so this observer runs after the CPUIDInfo singleton is destroyed.
  static const CpuinfoTeardownObserver observer{deinitialized};
  static_cast<void>(onnxruntime::CPUIDInfo::GetCPUIDInfo());
  return cpuinfo_get_processors() != nullptr && cpuinfo_get_processors_count() != 0;
}
