// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <windows.h>
#include <dbghelp.h>
#include <iostream>
#include <string>
#pragma comment(lib, "dbghelp.lib")

struct HMODULE_Deleter {
  typedef HMODULE pointer;
  void operator()(HMODULE h) { FreeLibrary(h); }
};

using ModulePtr = std::unique_ptr<HMODULE, HMODULE_Deleter>;

// If a DLL fails to load, this will try loading the DLL and then its dependencies recursively
// until it finds a missing file, then will report which file is missing and what the dependency
// chain is.
std::wstring DetermineLoadLibraryError(const wchar_t* filename_in) {
  std::wstring error(L"Error loading");

  std::wstring filename{filename_in};
  while (filename.size()) {
    error += std::wstring(L" \"") + filename + L"\"";

    ModulePtr hModule = ModulePtr{LoadLibraryExW(filename.c_str(), NULL, DONT_RESOLVE_DLL_REFERENCES)};
    if (!hModule) {
      error += L" which is missing.";
      return error;
    }

    // Get the address of the Import Directory
    ULONG size;
    PIMAGE_IMPORT_DESCRIPTOR importDesc = (PIMAGE_IMPORT_DESCRIPTOR)ImageDirectoryEntryToData(hModule.get(), TRUE, IMAGE_DIRECTORY_ENTRY_IMPORT, &size);
    if (!importDesc) {
      error += L" No import directory found.";  // This is unexpected, and I'm not sure how it could happen but we handle it just in case.
      return error;
    }

    // Iterate through the import descriptors to see which dependent DLL can't load
    filename.clear();
    for (; importDesc->Characteristics; importDesc++) {
      char* dllName = (char*)((BYTE*)(hModule.get()) + importDesc->Name);
      ModulePtr hDepModule{LoadLibrary(dllName)};
      if (!hDepModule) {
        filename = std::wstring(dllName, dllName + strlen(dllName));
        error += L" which depends on";
        break;
      }
    }
  }
  error += L" But no dependency issue could be determined.";
  return error;
}
