// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "cpuinfo.h"

int main() {
  for (int cycle = 0; cycle != 3; ++cycle) {
    if (!cpuinfo_initialize()) {
      std::cerr << "cpuinfo initialization failed in cycle " << cycle
                << "; check CPU detection diagnostics and host CPU metadata access.\n";
      return 1;
    }
    const bool acquired_again = cpuinfo_initialize();
    const auto* processor = cpuinfo_get_processor(0);
    const bool detected = acquired_again && processor != nullptr && processor->core != nullptr &&
                          processor->core->vendor != cpuinfo_vendor_unknown &&
                          cpuinfo_get_processors_count() > 0 && cpuinfo_get_cores_count() > 0;
    if (acquired_again) {
      cpuinfo_deinitialize();
    }
    const bool retained = cpuinfo_get_processors_count() > 0;
    cpuinfo_deinitialize();
    if (!detected || !retained) {
      std::cerr << "cpuinfo failed vendor/topology or reference-counted lifetime checks in cycle "
                << cycle << "; verify the pinned source, lifecycle patch, and host CPU support.\n";
      return 1;
    }
  }
  std::cout << "cpuinfo vendor/topology detection and three balanced reinitialization cycles passed.\n";
  return 0;
}
