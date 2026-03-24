// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/system_info.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace onnxruntime {
namespace test {

std::optional<uint64_t> GetTotalPhysicalMemoryBytes() {
#ifdef _WIN32
  MEMORYSTATUSEX mem_info = {};
  mem_info.dwLength = sizeof(mem_info);
  if (GlobalMemoryStatusEx(&mem_info)) {
    return static_cast<uint64_t>(mem_info.ullTotalPhys);
  }
#else
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGESIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
  }
#endif
  return std::nullopt;
}

}  // namespace test
}  // namespace onnxruntime
