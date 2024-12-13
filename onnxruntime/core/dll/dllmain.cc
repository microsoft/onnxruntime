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

// == workaround for delay loading of dependencies of onnxruntime.dll ==
//
// Problem:
//
// When onnxruntime.dll uses delay loading for its dependencies, the dependencies are loaded using LoadLibraryEx,
// which search the directory of process (.exe) instead of this library (onnxruntime.dll). This is a problem for
// usages of Node.js binding and python binding, because Windows will try to find the dependencies in the directory
// of node.exe or python.exe, which is not the directory of onnxruntime.dll.
//
// Solution:
//
// By using Win32 API `SetDefaultDllDirectories` and `AddDllDirectory`, we can modify the DLL search order to include
// the directory of onnxruntime.dll. This will make sure the dependencies are loaded from the directory of onnxruntime.dll
// when later calling LoadLibraryEx() without flags.
//
// See https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order#alternate-search-order-for-unpackaged-apps

#if defined(_MSC_VER) && defined(USE_WEBGPU)
#define USE_DELAYLOAD_WORKAROUND 1
#else
#define USE_DELAYLOAD_WORKAROUND 0
#endif

#if USE_DELAYLOAD_WORKAROUND
namespace {
DLL_DIRECTORY_COOKIE onnxruntime_dll_dir_cookie;
}
#endif

// dllmain.cpp : Defines the entry point for the DLL application.
BOOL APIENTRY DllMain(HMODULE hModule,
                      DWORD ul_reason_for_call,
                      LPVOID /*lpReserved*/
) {
  switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH: {
#if !USE_DELAYLOAD_WORKAROUND
      UNREFERENCED_PARAMETER(hModule);
#else
      onnxruntime_dll_dir_cookie = NULL;
      WCHAR path[MAX_PATH + 1];
      DWORD len = GetModuleFileNameW(hModule, path, MAX_PATH + 1);
      if (len == 0 || len > MAX_PATH) {
        // Failed to get the path of the current module. Skip adding DLL search directory.
        return TRUE;
      }

      // Remove the file name from the path
      while (len > 0 && path[len - 1] != L'\\') {
        --len;
      }
      if (len == 0) {
        // Seems not a valid path. Skip adding DLL search directory.
        return TRUE;
      }
      path[len] = L'\0';

      if (SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
        onnxruntime_dll_dir_cookie = AddDllDirectory(path);
      }
#endif  // USE_DELAYLOAD_WORKAROUND
      break;
    }
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
      break;
    case DLL_PROCESS_DETACH:
      // TODO: Don't do it when Protobuf_USE_STATIC_LIBS is OFF
      ::google::protobuf::ShutdownProtobufLibrary();

#if USE_DELAYLOAD_WORKAROUND
      if (onnxruntime_dll_dir_cookie != NULL) {
        RemoveDllDirectory(onnxruntime_dll_dir_cookie);
      }
#endif  // USE_DELAYLOAD_WORKAROUND
      break;
  }
  return TRUE;
}
