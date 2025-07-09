// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include "core/session/ort_env.h"

extern std::atomic<bool> g_is_shutting_down;

// dllmain.cc : Defines the entry point for the DLL application.
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
        g_is_shutting_down = true;
        // do not do cleanup if process termination scenario
      } else {
        // Cleanup protobuf library.
        // NOTE: it might be too early to do so, as all function local statics and global objects are not destroyed yet.
        ::google::protobuf::ShutdownProtobufLibrary();
      }
      break;
  }
  return TRUE;
}
