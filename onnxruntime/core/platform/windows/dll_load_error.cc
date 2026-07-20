// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#include <iostream>
#include <memory>
#include <string>
#include "dll_load_error.h"

struct HMODULE_Deleter {
  typedef HMODULE pointer;
  void operator()(HMODULE h) { FreeLibrary(h); }
};

using ModulePtr = std::unique_ptr<HMODULE, HMODULE_Deleter>;

// If a DLL fails to load, this will try loading the DLL and then its dependencies recursively
// until it finds a missing file, then will report which file is missing and what the dependency
// chain is.
std::wstring DetermineLoadLibraryError(const wchar_t* filename_in, DWORD flags) {
  std::wstring error(L"Error loading");

  std::wstring filename{filename_in};
  while (filename.size()) {
    error += std::wstring(L" \"") + filename + L"\"";

    // We use DONT_RESOLVE_DLL_REFERENCES instead of LOAD_LIBRARY_AS_DATAFILE because the latter will not process the import table
    // and will result in the IMAGE_IMPORT_DESCRIPTOR table names being uninitialized.
    ModulePtr hModule = ModulePtr{LoadLibraryExW(filename.c_str(), NULL, flags | DONT_RESOLVE_DLL_REFERENCES)};
    if (!hModule) {
      error += L" which is missing.";
      return error;
    }

    // Get the address of the Import Directory
    ULONG size{};
    IMAGE_IMPORT_DESCRIPTOR* import_desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(ImageDirectoryEntryToData(hModule.get(), TRUE, IMAGE_DIRECTORY_ENTRY_IMPORT, &size));
    if (!import_desc) {
      error += L" No import directory found.";  // This is unexpected, and I'm not sure how it could happen but we handle it just in case.
      return error;
    }

    // Iterate through the import descriptors to see which dependent DLL can't load
    filename.clear();
    flags = 0;  // Dependent libraries are relative, and flags like LOAD_WITH_ALTERED_SEARCH_PATH is undefined for those.
    for (; import_desc->Characteristics; import_desc++) {
      const char* dll_name = reinterpret_cast<const char*>(reinterpret_cast<const BYTE*>(hModule.get()) + import_desc->Name);

      // Convert the narrow (ANSI) DLL name to a wide string for the W API.
      // PE import table names are ANSI strings, so CP_ACP is the correct code page.
      int wide_len = MultiByteToWideChar(CP_ACP, 0, dll_name, -1, nullptr, 0);
      if (wide_len <= 0) {
        continue;  // Skip this entry if conversion fails.
      }
      std::wstring wide_dll_name(wide_len, L'\0');  // wide_len includes the null terminator
      int converted = MultiByteToWideChar(CP_ACP, 0, dll_name, -1, wide_dll_name.data(), wide_len);
      if (converted <= 0) {
        continue;  // Skip this entry if conversion fails.
      }
      wide_dll_name.resize(static_cast<size_t>(converted - 1));

      // Try to load the dependent DLL, and if it fails, we loop again with this as the DLL and we'll be one step closer to the missing file.
      ModulePtr hDepModule{LoadLibraryW(wide_dll_name.c_str())};
      if (!hDepModule) {
        filename = wide_dll_name;
        error += L" which depends on";
        break;
      }
    }
  }
  error += L" But no dependency issue could be determined.";
  return error;
}
