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
// The DLL DelayLoad hook is only enabled when:
// - The compiler is MSVC
// - at least one of USE_WEBGPU or USE_DML is defined
//
#if defined(_MSC_VER) && (defined(USE_WEBGPU) || defined(USE_DML))

#include <Windows.h>
#include <delayimp.h>
#include <stdlib.h>
#include <string>

namespace {

#define DEFINE_KNOWN_DLL(name) {#name ".dll", L#name L".dll"}

constexpr struct {
  const char* str;
  const wchar_t* wstr;
} known_dlls[] = {
#if defined(USE_WEBGPU)
    DEFINE_KNOWN_DLL(webgpu_dawn),
#endif
#if defined(USE_DML)
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
        DWORD pathLen = MAX_PATH;
        std::wstring path(pathLen, L'\0');
        HMODULE moduleHandle = nullptr;

        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&delay_load_hook), &moduleHandle);

        DWORD getModuleFileNameResult = GetModuleFileNameW(moduleHandle, const_cast<wchar_t*>(path.c_str()), pathLen);
        while (getModuleFileNameResult == 0 || getModuleFileNameResult == pathLen) {
          int ret = GetLastError();
          if (ret == ERROR_INSUFFICIENT_BUFFER && pathLen < 32768) {
            pathLen *= 2;
            path.resize(pathLen);
            getModuleFileNameResult = GetModuleFileNameW(moduleHandle, const_cast<wchar_t*>(path.c_str()), pathLen);
          } else {
            // Failed to get the path to onnxruntime.dll. In this case, we will just return NULL and let the system
            // search for the DLL in the default search order.
            return NULL;
          }
        }

        path.resize(path.rfind(L'\\') + 1);
        path.append(known_dlls[i].wstr);

        return FARPROC(LoadLibraryExW(path.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR));
      }
    }
  }
  return NULL;
}

extern "C" const PfnDliHook __pfnDliNotifyHook2 = delay_load_hook;

#endif
