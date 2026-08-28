// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "onnxruntime_c_api.h"

namespace {

// cpuinfo's Windows backend allocates its global topology data directly from the process heap.
struct ProcessHeapSnapshot {
  size_t busy_block_count = 0;
  size_t busy_bytes = 0;
};

struct ProcessHeapGrowth {
  int64_t busy_block_count = 0;
  int64_t busy_bytes = 0;
};

bool CaptureProcessHeapSnapshot(ProcessHeapSnapshot& snapshot) {
  HANDLE process_heap = GetProcessHeap();
  if (!HeapLock(process_heap)) {
    return false;
  }

  PROCESS_HEAP_ENTRY entry{};
  while (HeapWalk(process_heap, &entry)) {
    if ((entry.wFlags & PROCESS_HEAP_ENTRY_BUSY) != 0) {
      ++snapshot.busy_block_count;
      snapshot.busy_bytes += entry.cbData;
    }
  }

  const DWORD error = GetLastError();
  const bool unlocked = HeapUnlock(process_heap) != FALSE;
  return error == ERROR_NO_MORE_ITEMS && unlocked;
}

bool CheckStatus(const OrtApi& ort_api, OrtStatus* status) {
  if (status == nullptr) {
    return true;
  }

  std::cerr << "ONNX Runtime error: " << ort_api.GetErrorMessage(status) << std::endl;
  ort_api.ReleaseStatus(status);
  return false;
}

bool LoadAndUnload(bool query_hardware) {
  HMODULE ort_library = LoadLibraryW(L"onnxruntime.dll");
  if (ort_library == nullptr) {
    std::cerr << "LoadLibraryW failed with error " << GetLastError() << std::endl;
    return false;
  }

  using OrtGetApiBaseFunction = const OrtApiBase*(ORT_API_CALL*)();
  const auto ort_get_api_base =
      reinterpret_cast<OrtGetApiBaseFunction>(GetProcAddress(ort_library, "OrtGetApiBase"));
  if (ort_get_api_base == nullptr) {
    std::cerr << "GetProcAddress failed with error " << GetLastError() << std::endl;
    FreeLibrary(ort_library);
    return false;
  }

  const OrtApiBase* ort_api_base = ort_get_api_base();
  if (ort_api_base == nullptr) {
    std::cerr << "OrtGetApiBase returned null" << std::endl;
    FreeLibrary(ort_library);
    return false;
  }

  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  OrtEnv* env = nullptr;
  bool success = ort_api != nullptr;

  if (success) {
    success = CheckStatus(*ort_api, ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CpuinfoDlopenTest", &env));
  }
  if (success && query_hardware) {
    size_t num_devices = 0;
    success = CheckStatus(*ort_api, ort_api->GetNumHardwareDevices(env, &num_devices));
    if (success && num_devices == 0) {
      std::cerr << "ONNX Runtime reported no hardware devices" << std::endl;
      success = false;
    }
  }
  if (env != nullptr) {
    ort_api->ReleaseEnv(env);
  }

  if (!FreeLibrary(ort_library)) {
    std::cerr << "FreeLibrary failed with error " << GetLastError() << std::endl;
    return false;
  }

  if (GetModuleHandleW(L"onnxruntime.dll") != nullptr) {
    std::cerr << "onnxruntime.dll remained loaded after FreeLibrary" << std::endl;
    return false;
  }

  return success;
}

ProcessHeapGrowth operator-(const ProcessHeapSnapshot& after, const ProcessHeapSnapshot& before) {
  return {
      static_cast<int64_t>(after.busy_block_count) - static_cast<int64_t>(before.busy_block_count),
      static_cast<int64_t>(after.busy_bytes) - static_cast<int64_t>(before.busy_bytes),
  };
}

}  // namespace

int wmain() {
  constexpr size_t kWarmupCycles = 2;
  constexpr size_t kMeasuredCycles = 3;

  for (size_t cycle = 0; cycle < kWarmupCycles; ++cycle) {
    if (!LoadAndUnload(false) || !LoadAndUnload(true)) {
      return EXIT_FAILURE;
    }
  }

  ProcessHeapGrowth baseline_growth;
  ProcessHeapGrowth cpuinfo_growth;

  // CreateEnv must not initialize CPUIDInfo. GetNumHardwareDevices does so through Windows device discovery.
  for (size_t cycle = 0; cycle < kMeasuredCycles; ++cycle) {
    ProcessHeapSnapshot before_baseline;
    ProcessHeapSnapshot after_baseline;
    ProcessHeapSnapshot after_cpuinfo;
    if (!CaptureProcessHeapSnapshot(before_baseline)) {
      std::cerr << "Failed to capture process heap before baseline cycle " << cycle << std::endl;
      return EXIT_FAILURE;
    }
    if (!LoadAndUnload(false) || !CaptureProcessHeapSnapshot(after_baseline)) {
      std::cerr << "Failed to measure baseline DLL cycle " << cycle << std::endl;
      return EXIT_FAILURE;
    }
    if (!LoadAndUnload(true) || !CaptureProcessHeapSnapshot(after_cpuinfo)) {
      std::cerr << "Failed to measure cpuinfo DLL cycle " << cycle << std::endl;
      return EXIT_FAILURE;
    }

    const ProcessHeapGrowth baseline_cycle = after_baseline - before_baseline;
    const ProcessHeapGrowth cpuinfo_cycle = after_cpuinfo - after_baseline;
    baseline_growth.busy_block_count += baseline_cycle.busy_block_count;
    baseline_growth.busy_bytes += baseline_cycle.busy_bytes;
    cpuinfo_growth.busy_block_count += cpuinfo_cycle.busy_block_count;
    cpuinfo_growth.busy_bytes += cpuinfo_cycle.busy_bytes;
  }

  constexpr int64_t kBlockTolerance = 2;
  constexpr int64_t kByteTolerance = 1024;
  if (cpuinfo_growth.busy_block_count > baseline_growth.busy_block_count + kBlockTolerance &&
      cpuinfo_growth.busy_bytes > baseline_growth.busy_bytes + kByteTolerance) {
    std::cerr << "Hardware discovery retained additional process-heap allocations across DLL unload cycles:"
              << " baseline blocks/bytes " << baseline_growth.busy_block_count << "/"
              << baseline_growth.busy_bytes
              << ", cpuinfo blocks/bytes " << cpuinfo_growth.busy_block_count << "/"
              << cpuinfo_growth.busy_bytes << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
