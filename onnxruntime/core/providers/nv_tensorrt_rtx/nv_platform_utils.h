// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include "core/common/path_string.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// The filename extension for a shared library is different per platform
#ifdef _WIN32
#define LIBRARY_PREFIX
#define LIBRARY_EXTENSION ORT_TSTR(".dll")
#elif defined(__APPLE__)
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".dylib"
#else
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".so"
#endif

namespace onnxruntime {
inline PathString GetEPLibraryDirectory() {
#ifdef _WIN32
  HMODULE hModule = NULL;
  // Get handle to the DLL executing this code
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(GetEPLibraryDirectory),
                          &hModule)) {
    return PathString();
  }

  wchar_t buffer[MAX_PATH];
  DWORD len = GetModuleFileNameW(hModule, buffer, MAX_PATH);
  if (len == 0 || len >= MAX_PATH) {
    return PathString();
  }

  std::wstring path(buffer);
  size_t lastSlash = path.find_last_of(L"\\/");
  if (lastSlash != std::wstring::npos) {
    return PathString(path.substr(0, lastSlash + 1));
  }
  return PathString();
#else
  // Linux and other Unix-like platforms
  Dl_info dl_info;

  if (dladdr((void*)&GetEPLibraryDirectory, &dl_info) == 0 || dl_info.dli_fname == nullptr) {
    return PathString();
  }

  std::string so_path(dl_info.dli_fname);
  size_t last_slash = so_path.find_last_of('/');
  if (last_slash != std::string::npos) {
    return PathString(so_path.substr(0, last_slash + 1));
  }
  return PathString();
#endif
}
}  // namespace onnxruntime
