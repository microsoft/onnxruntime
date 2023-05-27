// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#include "common.h"
#include "windows.h"

void LoadDirectMLDll(Napi::Env env)
{
  DWORD pathLen = MAX_PATH;
  wchar_t* path = new wchar_t[pathLen];
  HMODULE moduleHandle = nullptr;

  GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                 GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCSTR>(&LoadDirectMLDll), &moduleHandle);

  DWORD getModuleFileNameResult = GetModuleFileNameW(moduleHandle, path, pathLen);
  while (getModuleFileNameResult == 0 || getModuleFileNameResult == pathLen) {
    delete[] path;

    pathLen = pathLen * 2;
    int ret = GetLastError();
    if (ret == ERROR_INSUFFICIENT_BUFFER && pathLen < 32768) {
      path = new wchar_t[pathLen];
      getModuleFileNameResult = GetModuleFileNameW(moduleHandle, path, pathLen);
    } else {
      ORT_NAPI_THROW_ERROR(env, "Failed getting path to load DirectML.dll, error code: ", ret);
    }
  }

  // assume binding name will be always longer (onnxruntime_binding.node)
  wchar_t* lastBackslash = wcsrchr(path, L'\\');
  *lastBackslash = L'\0';
  wcscat_s(path, pathLen, L"\\DirectML.dll");
  auto libraryLoadResult = LoadLibraryW(path);
  delete[] path;

  if (!libraryLoadResult) {
    int ret = GetLastError();
    ORT_NAPI_THROW_ERROR(env, "Failed loading bundled DirectML.dll, error code: ", ret);
  }
}
#endif
