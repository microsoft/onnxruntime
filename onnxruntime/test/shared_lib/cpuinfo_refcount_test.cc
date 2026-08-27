// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cpuinfo.h>

#include <cstdlib>
#include <iostream>

int main() {
  if (!cpuinfo_initialize()) {
    std::cerr << "The first cpuinfo_initialize call failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (!cpuinfo_initialize()) {
    std::cerr << "The second cpuinfo_initialize call failed" << std::endl;
    cpuinfo_deinitialize();
    return EXIT_FAILURE;
  }

  const uint32_t processor_count = cpuinfo_get_processors_count();
  const cpuinfo_processor* processors = cpuinfo_get_processors();
  if (processor_count == 0 || processors == nullptr) {
    std::cerr << "cpuinfo did not expose valid processor data" << std::endl;
    cpuinfo_deinitialize();
    cpuinfo_deinitialize();
    return EXIT_FAILURE;
  }

  cpuinfo_deinitialize();

  const bool first_consumer_release_preserved_state =
      cpuinfo_get_processors_count() == processor_count &&
      cpuinfo_get_processors() == processors;

  cpuinfo_deinitialize();

  if (!first_consumer_release_preserved_state) {
    std::cerr << "cpuinfo released shared state while another consumer was still active" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
