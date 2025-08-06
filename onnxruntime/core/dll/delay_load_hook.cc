// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
// By using the delay load hook `__pfnDliNotifyHook2`, we can intervene the loading procedure by loading from an
// absolute path. The absolute path is constructed by appending the name of the DLL to load to the directory of
// onnxruntime.dll. This way, we can ensure that the dependencies are loaded from the same directory as onnxruntime.dll.
//
// See also:
// - https://learn.microsoft.com/en-us/cpp/build/reference/understanding-the-helper-function?view=msvc-170#structure-and-constant-definitions
// - https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order#alternate-search-order-for-unpackaged-apps
//
// The DLL DelayLoad hook is only enabled when the compiler is MSVC and at least one of the following is True:
// - both USE_WEBGPU and BUILD_DAWN_MONOLITHIC_LIBRARY are defined
// - USE_DML is defined
//
#if defined(USE_WEBGPU) && defined(BUILD_DAWN_MONOLITHIC_LIBRARY)
#define ORT_DELAY_LOAD_WEBGPU_DAWN_DLL 1
#else
#define ORT_DELAY_LOAD_WEBGPU_DAWN_DLL 0
#endif
#if defined(USE_DML)
#define ORT_DELAY_LOAD_DIRECTML_DLL 1
#else
#define ORT_DELAY_LOAD_DIRECTML_DLL 0
#endif
#if defined(_MSC_VER) && (ORT_DELAY_LOAD_WEBGPU_DAWN_DLL || ORT_DELAY_LOAD_DIRECTML_DLL)

#include <Windows.h>
#include <delayimp.h>
#include <stdlib.h>
#include <string>

#include "core/platform/env.h"

namespace {

#define DEFINE_KNOWN_DLL(name) {#name ".dll", L## #name L".dll"}

constexpr struct {
  const char* str;
  const wchar_t* wstr;
} known_dlls[] = {
#if ORT_DELAY_LOAD_WEBGPU_DAWN_DLL
    DEFINE_KNOWN_DLL(webgpu_dawn),
#endif
#if ORT_DELAY_LOAD_DIRECTML_DLL
    DEFINE_KNOWN_DLL(DirectML),
#endif
};
}  // namespace

FARPROC WINAPI delay_load_hook(unsigned dliNotify, PDelayLoadInfo pdli) {
  if (dliNotify == dliNotePreLoadLibrary) {
    for (size_t i = 0; i < _countof(known_dlls); ++i) {
      if (_stricmp(pdli->szDll, known_dlls[i].str) == 0) {
        // Try to load the DLL from the same directory as onnxruntime.dll

        // First, get the path to onnxruntime.dll
        auto path = onnxruntime::Env::Default().GetRuntimePath();
        if (path.empty()) {
          // Failed to get the path to onnxruntime.dll. In this case, we will just return NULL and let the system
          // search for the DLL in the default search order.
          return NULL;
        }

        // Append the name of the DLL. Now `path` is the absolute path to the DLL to load.
        path.append(known_dlls[i].wstr);

        // Load the DLL
        return FARPROC(LoadLibraryExW(path.c_str(), NULL,
                                      LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR));
      }
    }
  }
  return NULL;
}

extern "C" const PfnDliHook __pfnDliNotifyHook2 = delay_load_hook;

#endif
