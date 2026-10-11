// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstdlib>
#include <iostream>

#ifndef ORT_CPUINFO_DLOPEN_TEST_LIBRARY
#error ORT_CPUINFO_DLOPEN_TEST_LIBRARY must name the test DLL.
#endif

int wmain() {
  HMODULE library = LoadLibraryW(ORT_CPUINFO_DLOPEN_TEST_LIBRARY);
  if (library == nullptr) {
    std::cerr << "LoadLibraryW failed with error " << GetLastError() << std::endl;
    return EXIT_FAILURE;
  }

  using GetCpuinfoAllocation = const void* (*)();
  const auto get_cpuinfo_allocation = reinterpret_cast<GetCpuinfoAllocation>(
      GetProcAddress(library, "OrtGetCpuinfoAllocationForTesting"));
  if (get_cpuinfo_allocation == nullptr) {
    std::cerr << "GetProcAddress failed with error " << GetLastError() << std::endl;
    FreeLibrary(library);
    return EXIT_FAILURE;
  }

  const void* allocation = get_cpuinfo_allocation();
  HANDLE process_heap = GetProcessHeap();
  if (allocation == nullptr || !HeapValidate(process_heap, 0, allocation)) {
    std::cerr << "cpuinfo did not return a valid process-heap allocation" << std::endl;
    FreeLibrary(library);
    return EXIT_FAILURE;
  }

  if (!FreeLibrary(library)) {
    std::cerr << "FreeLibrary failed with error " << GetLastError() << std::endl;
    return EXIT_FAILURE;
  }

  if (HeapValidate(process_heap, 0, allocation)) {
    std::cerr << "cpuinfo allocation remained valid after the CPUIDInfo test DLL was unloaded" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
