// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#include "common.h"
#include "windows.h"

void LoadDirectMLDll(Napi::Env env) {
  DWORD pathLen = MAX_PATH;
  std::wstring path(pathLen, L'\0');
  HMODULE moduleHandle = nullptr;

  GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCSTR>(&LoadDirectMLDll), &moduleHandle);

  DWORD getModuleFileNameResult = GetModuleFileNameW(moduleHandle, const_cast<wchar_t *>(path.c_str()), pathLen);
  while (getModuleFileNameResult == 0 || getModuleFileNameResult == pathLen) {
    int ret = GetLastError();
    if (ret == ERROR_INSUFFICIENT_BUFFER && pathLen < 32768) {
      pathLen *= 2;
      path.resize(pathLen);
      getModuleFileNameResult = GetModuleFileNameW(moduleHandle, const_cast<wchar_t *>(path.c_str()), pathLen);
    } else {
      ORT_NAPI_THROW_ERROR(env, "Failed getting path to load DirectML.dll, error code: ", ret);
    }
  }

  path.resize(path.rfind(L'\\') + 1);
  path.append(L"DirectML.dll");
  HMODULE libraryLoadResult = LoadLibraryW(path.c_str());

  if (!libraryLoadResult) {
    int ret = GetLastError();
    ORT_NAPI_THROW_ERROR(env, "Failed loading bundled DirectML.dll, error code: ", ret);
  }
}
#endif
