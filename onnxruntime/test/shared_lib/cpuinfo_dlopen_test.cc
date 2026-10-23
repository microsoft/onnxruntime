// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstdlib>
#include <iostream>

#ifndef ORT_CPUINFO_DLOPEN_TEST_LIBRARY
#error ORT_CPUINFO_DLOPEN_TEST_LIBRARY must name the test DLL.
#endif

namespace {
// This storage must remain valid through DLL teardown, including process-exit error paths.
bool cpuinfo_deinitialized = false;
}  // namespace

int wmain() {
  HMODULE library = LoadLibraryW(ORT_CPUINFO_DLOPEN_TEST_LIBRARY);
  if (library == nullptr) {
    std::cerr << "LoadLibraryW failed with error " << GetLastError() << std::endl;
    return EXIT_FAILURE;
  }

  using InitializeCpuinfo = bool (*)(bool*);
  const auto initialize_cpuinfo = reinterpret_cast<InitializeCpuinfo>(
      GetProcAddress(library, "OrtInitializeCpuinfoForTesting"));
  if (initialize_cpuinfo == nullptr) {
    std::cerr << "GetProcAddress failed with error " << GetLastError() << std::endl;
    FreeLibrary(library);
    return EXIT_FAILURE;
  }

  if (!initialize_cpuinfo(&cpuinfo_deinitialized)) {
    std::cerr << "cpuinfo did not expose valid processor data" << std::endl;
    FreeLibrary(library);
    return EXIT_FAILURE;
  }

  if (!FreeLibrary(library)) {
    std::cerr << "FreeLibrary failed with error " << GetLastError() << std::endl;
    return EXIT_FAILURE;
  }

  if (!cpuinfo_deinitialized) {
    std::cerr << "cpuinfo remained initialized after the CPUIDInfo test DLL was unloaded" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
