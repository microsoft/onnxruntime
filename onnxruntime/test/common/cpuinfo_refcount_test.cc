// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cpuinfo.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

namespace {

bool HasValidCpuinfoState() {
  return cpuinfo_get_processors_count() != 0 &&
         cpuinfo_get_processors() != nullptr &&
         cpuinfo_get_processor(0) != nullptr;
}

bool TestSequentialConsumers() {
  if (!cpuinfo_initialize()) {
    std::cerr << "The first cpuinfo_initialize call failed" << std::endl;
    return false;
  }

  if (!cpuinfo_initialize()) {
    std::cerr << "The second cpuinfo_initialize call failed" << std::endl;
    cpuinfo_deinitialize();
    return false;
  }

  const uint32_t processor_count = cpuinfo_get_processors_count();
  const cpuinfo_processor* processors = cpuinfo_get_processors();
  if (processor_count == 0 || processors == nullptr) {
    std::cerr << "cpuinfo did not expose valid processor data" << std::endl;
    cpuinfo_deinitialize();
    cpuinfo_deinitialize();
    return false;
  }

  cpuinfo_deinitialize();

  const bool first_consumer_release_preserved_state =
      cpuinfo_get_processors_count() == processor_count &&
      cpuinfo_get_processors() == processors;

  cpuinfo_deinitialize();

  if (!first_consumer_release_preserved_state) {
    std::cerr << "cpuinfo released shared state while another consumer was still active" << std::endl;
    return false;
  }

  if (!cpuinfo_initialize()) {
    std::cerr << "cpuinfo failed to reinitialize after the final consumer released it" << std::endl;
    return false;
  }
  cpuinfo_deinitialize();

  return true;
}

bool TestConcurrentConsumers() {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kIterations = 100;
  std::atomic<size_t> initialized_count{0};
  std::atomic<size_t> validated_count{0};
  std::atomic<size_t> deinitialized_count{0};
  std::atomic<bool> failed{false};

  const auto consumer = [&]() {
    for (size_t iteration = 0; iteration < kIterations; ++iteration) {
      const bool initialized = cpuinfo_initialize();
      if (!initialized) {
        failed = true;
      }

      const size_t expected_count = (iteration + 1) * kThreadCount;
      ++initialized_count;
      while (initialized_count < expected_count) {
        std::this_thread::yield();
      }

      if (initialized && !HasValidCpuinfoState()) {
        failed = true;
      }

      ++validated_count;
      while (validated_count < expected_count) {
        std::this_thread::yield();
      }

      if (initialized) {
        cpuinfo_deinitialize();
      }

      ++deinitialized_count;
      while (deinitialized_count < expected_count) {
        std::this_thread::yield();
      }
    }
  };

  std::vector<std::thread> consumers;
  consumers.reserve(kThreadCount);
  for (size_t thread = 0; thread < kThreadCount; ++thread) {
    consumers.emplace_back(consumer);
  }
  for (std::thread& consumer_thread : consumers) {
    consumer_thread.join();
  }

  if (failed) {
    std::cerr << "cpuinfo failed during concurrent initialization and deinitialization" << std::endl;
    return false;
  }

  if (!cpuinfo_initialize() || !HasValidCpuinfoState()) {
    std::cerr << "cpuinfo failed to reinitialize after concurrent use" << std::endl;
    return false;
  }
  cpuinfo_deinitialize();

  return true;
}

}  // namespace

int main() {
  if (!TestSequentialConsumers() || !TestConcurrentConsumers()) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
