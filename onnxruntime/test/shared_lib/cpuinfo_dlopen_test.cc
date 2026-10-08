// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstdlib>
#include <iostream>

#ifndef ORT_CPUINFO_DLOPEN_TEST_LIBRARY
#error ORT_CPUINFO_DLOPEN_TEST_LIBRARY must name the test DLL.
#endif

namespace {

bool TestUnload(bool retain_reference) {
  HMODULE library = LoadLibraryW(ORT_CPUINFO_DLOPEN_TEST_LIBRARY);
  if (library == nullptr) {
    std::cerr << "LoadLibraryW failed with error " << GetLastError() << std::endl;
    return false;
  }

  using GetCpuinfoAllocation = const void* (*)(int*, bool);
  const auto get_cpuinfo_allocation = reinterpret_cast<GetCpuinfoAllocation>(
      GetProcAddress(library, "OrtGetCpuinfoAllocationForTesting"));
  if (get_cpuinfo_allocation == nullptr) {
    std::cerr << "GetProcAddress failed with error " << GetLastError() << std::endl;
    FreeLibrary(library);
    return false;
  }

  int unload_state = -1;
  const void* allocation = get_cpuinfo_allocation(&unload_state, retain_reference);
  HANDLE process_heap = GetProcessHeap();
  if (allocation == nullptr || !HeapValidate(process_heap, 0, allocation)) {
    std::cerr << "cpuinfo did not return a valid process-heap allocation" << std::endl;
    FreeLibrary(library);
    return false;
  }

  if (!FreeLibrary(library)) {
    std::cerr << "FreeLibrary failed with error " << GetLastError() << std::endl;
    return false;
  }

  // A freed heap address can be reused during unload. Check the DLL's ownership state instead.
  const int expected_state = retain_reference ? 1 : 0;
  if (unload_state != expected_state) {
    std::cerr << "cpuinfo state after CPUIDInfo destruction was " << unload_state
              << ", expected " << expected_state << " (retained consumer: " << retain_reference << ")" << std::endl;
    return false;
  }

  return true;
}

}  // namespace

int wmain() {
  return TestUnload(false) && TestUnload(true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
