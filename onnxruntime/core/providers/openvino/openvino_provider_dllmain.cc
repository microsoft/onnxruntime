// Copyright (c) Intel Corporation.
// Licensed under the MIT License.
#ifdef _WIN32

#include <Windows.h>
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#endif
#include <google/protobuf/message_lite.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include <atomic>

// Reuse the global shutdown indicator (do NOT set it here; that is owned by the core DLL).
extern std::atomic<bool> g_is_shutting_down;

// NOTE:
// This DllMain exists because the OpenVINO provider DLL statically links protobuf independently
// of the core onnxruntime DLL. The core DLL's DllMain won't clean up this copy.
// We perform protobuf shutdown on dynamic unload, and (optionally) during process termination
// when memory leak checking is enabled.
BOOL APIENTRY DllMain(HMODULE /*hModule*/,
                      DWORD ul_reason_for_call,
                      LPVOID lpvReserved) {
  switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
      break;
    case DLL_PROCESS_DETACH:
      // Windows API doc says: "When handling DLL_PROCESS_DETACH, a DLL should free resources such as heap memory only if the DLL is being unloaded dynamically"
      if (lpvReserved != nullptr) {
        // Process termination. Normally skipped for speed/safety,
        // but in leak-check builds we reclaim protobuf heap.
#if defined(ONNXRUNTIME_ENABLE_MEMLEAK_CHECK)
        ::google::protobuf::ShutdownProtobufLibrary();
#endif
      } else {
        // Dynamic unload: safe to clean up.
        ::google::protobuf::ShutdownProtobufLibrary();
      }
      break;
  }
  return TRUE;
}

#endif  // defined(_WIN32)
