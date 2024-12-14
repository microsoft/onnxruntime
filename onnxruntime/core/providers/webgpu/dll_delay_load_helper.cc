// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/dll_delay_load_helper.h"

#if defined(_WIN32) && defined(_MSC_VER) && !defined(__EMSCRIPTEN__)

#include <Windows.h>
#include <delayimp.h>
#include <stdlib.h>
#include <string>
#include <mutex>

namespace onnxruntime {
namespace webgpu {

namespace {

// Get the directory of the current DLL (usually it's onnxruntime.dll).
std::wstring GetCurrentDllDir() {
  DWORD pathLen = MAX_PATH;
  std::wstring path(pathLen, L'\0');
  HMODULE moduleHandle = nullptr;

  GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                     reinterpret_cast<LPCWSTR>(&GetCurrentDllDir), &moduleHandle);

  DWORD getModuleFileNameResult = GetModuleFileNameW(moduleHandle, const_cast<wchar_t*>(path.c_str()), pathLen);
  while (getModuleFileNameResult == 0 || getModuleFileNameResult == pathLen) {
    int ret = GetLastError();
    if (ret == ERROR_INSUFFICIENT_BUFFER && pathLen < 32768) {
      pathLen *= 2;
      path.resize(pathLen);
      getModuleFileNameResult = GetModuleFileNameW(moduleHandle, const_cast<wchar_t*>(path.c_str()), pathLen);
    } else {
      // Failed to get the path to onnxruntime.dll. Returns an empty string.
      return std::wstring{};
    }
  }
  path.resize(path.rfind(L'\\') + 1);
  return path;
}

std::once_flag run_once_before_load_deps_mutex;
std::once_flag run_once_after_load_deps_mutex;
bool dll_dir_set = false;

}  // namespace

DllDelayLoadHelper::DllDelayLoadHelper() {
  // Setup DLL search directory
  std::call_once(run_once_before_load_deps_mutex, []() {
    std::wstring path = GetCurrentDllDir();
    if (!path.empty()) {
      SetDllDirectoryW(path.c_str());
      dll_dir_set = true;
    }
  });
}

DllDelayLoadHelper::~DllDelayLoadHelper() {
  // Restore DLL search directory
  std::call_once(run_once_after_load_deps_mutex, []() {
    if (dll_dir_set) {
      SetDllDirectoryW(NULL);
    }
  });
}

}  // namespace webgpu
}  // namespace onnxruntime

#else  // defined(_WIN32) && defined(_MSC_VER) && !defined(__EMSCRIPTEN__)

namespace onnxruntime {
namespace webgpu {

DllDelayLoadHelper::DllDelayLoadHelper() {
}

DllDelayLoadHelper::~DllDelayLoadHelper() {
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif
