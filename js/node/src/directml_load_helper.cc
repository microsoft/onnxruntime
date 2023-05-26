// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#include "common.h"
#include "windows.h"

extern "C" __declspec(dllexport) void ExportFunction()
{
}

std::string GetModuleDirectory()
{
  char path[MAX_PATH];
  HMODULE hm = NULL;

  if (!GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         (LPCSTR) &ExportFunction, &hm))
  {
    int ret = GetLastError();
    fprintf(stderr, "GetModuleHandleEx failed, error = %d\n", ret);
  }

  if (!GetModuleFileName(hm, path, sizeof(path)))
  {
    int ret = GetLastError();
    fprintf(stderr, "GetModuleFileName failed, error = %d\n", ret);
  }

  std::string::size_type pos = std::string(path).find_last_of("\\/");
  return std::string(path).substr(0, pos);
}

void LoadDirectMLDll()
{
  std::string module_dir = GetModuleDirectory();
  std::string dll_path = module_dir + "\\DirectML.dll";
  LoadLibrary(dll_path.c_str());
}
#endif
