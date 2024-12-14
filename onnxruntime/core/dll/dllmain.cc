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

#if defined(_MSC_VER) && (defined(USE_WEBGPU) || defined(USE_DML))
#define USE_DELAYLOAD_WORKAROUND 1
#else
#define USE_DELAYLOAD_WORKAROUND 0
#endif

#if USE_DELAYLOAD_WORKAROUND
#include <delayimp.h>
namespace {

constexpr const char* known_dlls[] = {
#if defined(USE_WEBGPU)
    "webgpu_dawn.dll",
#endif
#if defined(USE_DML)
    "DirectML.dll",
#endif
};

}  // namespace
#endif

// dllmain.cpp : Defines the entry point for the DLL application.
BOOL APIENTRY DllMain(HMODULE /*hModule*/,
                      DWORD ul_reason_for_call,
                      LPVOID /*lpReserved*/
) {
  switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
      break;
    case DLL_PROCESS_DETACH:
      // TODO: Don't do it when Protobuf_USE_STATIC_LIBS is OFF
      ::google::protobuf::ShutdownProtobufLibrary();
      break;
  }
  return TRUE;
}

#if USE_DELAYLOAD_WORKAROUND

FARPROC WINAPI delay_load_workaround_hook(unsigned dliNotify, PDelayLoadInfo pdli) {
  if (dliNotify == dliNotePreLoadLibrary) {
    return FARPROC(LoadLibraryExA(pdli->szDll, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR));
  }
  return NULL;
}

extern "C" const PfnDliHook __pfnDliNotifyHook2 = delay_load_workaround_hook;
#endif
