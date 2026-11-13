// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstdlib>
#include <iostream>

#ifndef ORT_CPUINFO_DLOPEN_TEST_LIBRARY
#error ORT_CPUINFO_DLOPEN_TEST_LIBRARY must name the test DLL.
#endif

int wmain() {
  // A retained module may run its observer only at process exit.
  static bool allocation_freed = false;
  for (int cycle = 0; cycle < 16; ++cycle) {
    HMODULE library = LoadLibraryW(ORT_CPUINFO_DLOPEN_TEST_LIBRARY);
    if (library == nullptr) {
      std::cerr << "LoadLibraryW failed with error " << GetLastError() << std::endl;
      return EXIT_FAILURE;
    }

    using GetCpuinfoAllocation = const void* (*)(bool*);
    const auto get_cpuinfo_allocation = reinterpret_cast<GetCpuinfoAllocation>(
        GetProcAddress(library, "OrtGetCpuinfoAllocationForTesting"));
    if (get_cpuinfo_allocation == nullptr) {
      std::cerr << "GetProcAddress failed with error " << GetLastError() << std::endl;
      FreeLibrary(library);
      return EXIT_FAILURE;
    }

    const void* allocation = get_cpuinfo_allocation(&allocation_freed);
    if (allocation == nullptr || !HeapValidate(GetProcessHeap(), 0, allocation)) {
      std::cerr << "cpuinfo did not return a valid process-heap allocation" << std::endl;
      FreeLibrary(library);
      return EXIT_FAILURE;
    }

    if (!FreeLibrary(library)) {
      std::cerr << "FreeLibrary failed with error " << GetLastError() << std::endl;
      return EXIT_FAILURE;
    }

    if (!allocation_freed) {
      std::cerr << "cpuinfo allocation was not freed by the CPUIDInfo test DLL destructor in cycle "
                << cycle << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
