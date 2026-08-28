// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "onnxruntime_c_api.h"

namespace {

// cpuinfo's Windows backend allocates its global topology data directly from the process heap.
struct ProcessHeapSnapshot {
  size_t busy_block_count = 0;
  size_t busy_bytes = 0;
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

bool LoadQueryHardwareAndUnload() {
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
  OrtSessionOptions* session_options = nullptr;
  OrtSession* session = nullptr;
  bool success = ort_api != nullptr;

  if (success) {
    success = CheckStatus(*ort_api, ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CpuinfoDlopenTest", &env));
  }
  if (success) {
    size_t num_devices = 0;
    success = CheckStatus(*ort_api, ort_api->GetNumHardwareDevices(env, &num_devices));
    if (success && num_devices == 0) {
      std::cerr << "ONNX Runtime reported no hardware devices" << std::endl;
      success = false;
    }
  }
#if defined(ORT_CPUINFO_DLOPEN_TEST_USE_XNNPACK)
  if (success) {
    success = CheckStatus(*ort_api, ort_api->CreateSessionOptions(&session_options));
  }
  if (success) {
    success = CheckStatus(
        *ort_api,
        ort_api->SessionOptionsAppendExecutionProvider(session_options, "XNNPACK", nullptr, nullptr, 0));
  }
  if (success) {
    success = CheckStatus(
        *ort_api,
        ort_api->CreateSession(env, L"testdata\\matmul_1.onnx", session_options, &session));
  }
#endif

  if (session != nullptr) {
    ort_api->ReleaseSession(session);
  }
  if (session_options != nullptr) {
    ort_api->ReleaseSessionOptions(session_options);
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

}  // namespace

int wmain() {
  constexpr size_t kWarmupCycles = 2;
  constexpr size_t kMeasuredCycles = 3;

  for (size_t cycle = 0; cycle < kWarmupCycles; ++cycle) {
    if (!LoadQueryHardwareAndUnload()) {
      return EXIT_FAILURE;
    }
  }

  std::array<ProcessHeapSnapshot, kMeasuredCycles + 1> snapshots;
  if (!CaptureProcessHeapSnapshot(snapshots[0])) {
    std::cerr << "Failed to capture the initial process heap snapshot" << std::endl;
    return EXIT_FAILURE;
  }

  for (size_t cycle = 0; cycle < kMeasuredCycles; ++cycle) {
    if (!LoadQueryHardwareAndUnload()) {
      return EXIT_FAILURE;
    }
    if (!CaptureProcessHeapSnapshot(snapshots[cycle + 1])) {
      std::cerr << "Failed to capture process heap snapshot " << cycle + 1 << std::endl;
      return EXIT_FAILURE;
    }
  }

  bool block_count_grew_each_cycle = true;
  bool allocated_bytes_grew_each_cycle = true;
  // Ignore one-time loader caching and detect the repeated growth caused by unreleased cpuinfo globals.
  for (size_t cycle = 0; cycle < kMeasuredCycles; ++cycle) {
    block_count_grew_each_cycle =
        block_count_grew_each_cycle &&
        snapshots[cycle + 1].busy_block_count > snapshots[cycle].busy_block_count;
    allocated_bytes_grew_each_cycle =
        allocated_bytes_grew_each_cycle &&
        snapshots[cycle + 1].busy_bytes > snapshots[cycle].busy_bytes;
  }

  if (block_count_grew_each_cycle && allocated_bytes_grew_each_cycle) {
    std::cerr << "Process heap grew after every ONNX Runtime DLL load/unload cycle:"
              << " blocks " << snapshots[0].busy_block_count << " -> "
              << snapshots[kMeasuredCycles].busy_block_count
              << ", bytes " << snapshots[0].busy_bytes << " -> "
              << snapshots[kMeasuredCycles].busy_bytes << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
