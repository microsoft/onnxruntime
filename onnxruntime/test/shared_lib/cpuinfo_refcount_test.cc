// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cpuinfo.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#if defined(ORT_CPUINFO_TEST_USE_XNNPACK)
#include <xnnpack.h>
#endif

#if defined(ORT_CPUINFO_TEST_HAS_INTERNAL_STATE)
constexpr int kCpuinfoCacheLevelCount = 5;

extern "C" {
extern bool cpuinfo_is_initialized;
extern struct cpuinfo_processor* cpuinfo_processors;
extern struct cpuinfo_core* cpuinfo_cores;
extern struct cpuinfo_cluster* cpuinfo_clusters;
extern struct cpuinfo_package* cpuinfo_packages;
extern struct cpuinfo_cache* cpuinfo_cache[kCpuinfoCacheLevelCount];
extern uint32_t cpuinfo_processors_count;
extern uint32_t cpuinfo_cores_count;
extern uint32_t cpuinfo_clusters_count;
extern uint32_t cpuinfo_packages_count;
extern uint32_t cpuinfo_cache_count[kCpuinfoCacheLevelCount];
extern uint32_t cpuinfo_max_cache_size;
}
#endif

namespace {

bool HasValidCpuinfoState() {
  return cpuinfo_get_processors_count() != 0 &&
         cpuinfo_get_processors() != nullptr &&
         cpuinfo_get_processor(0) != nullptr;
}

bool IsCpuinfoDeinitialized() {
#if defined(ORT_CPUINFO_TEST_HAS_INTERNAL_STATE)
  if (cpuinfo_is_initialized ||
      cpuinfo_processors != nullptr ||
      cpuinfo_cores != nullptr ||
      cpuinfo_clusters != nullptr ||
      cpuinfo_packages != nullptr ||
      cpuinfo_processors_count != 0 ||
      cpuinfo_cores_count != 0 ||
      cpuinfo_clusters_count != 0 ||
      cpuinfo_packages_count != 0 ||
      cpuinfo_max_cache_size != 0) {
    return false;
  }

  for (int level = 0; level < kCpuinfoCacheLevelCount; ++level) {
    if (cpuinfo_cache[level] != nullptr || cpuinfo_cache_count[level] != 0) {
      return false;
    }
  }

  return true;
#else
  return true;
#endif
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

  if (!IsCpuinfoDeinitialized()) {
    std::cerr << "cpuinfo remained initialized after the final consumer released it" << std::endl;
    return false;
  }

  if (!first_consumer_release_preserved_state) {
    std::cerr << "cpuinfo released shared state while another consumer was still active" << std::endl;
    return false;
  }

  if (!cpuinfo_initialize()) {
    std::cerr << "cpuinfo failed to reinitialize after the final consumer released it" << std::endl;
    return false;
  }
  cpuinfo_deinitialize();

  if (!IsCpuinfoDeinitialized()) {
    std::cerr << "cpuinfo remained initialized after the reinitialized consumer released it" << std::endl;
    return false;
  }

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

  if (!IsCpuinfoDeinitialized()) {
    std::cerr << "cpuinfo remained initialized after concurrent use" << std::endl;
    return false;
  }

  return true;
}

#if defined(ORT_CPUINFO_TEST_USE_XNNPACK)
bool TestXnnpackReleasesCpuinfo() {
  if (xnn_initialize(nullptr) != xnn_status_success) {
    std::cerr << "XNNPACK initialization failed" << std::endl;
    return false;
  }

  if (!IsCpuinfoDeinitialized()) {
    std::cerr << "XNNPACK retained a cpuinfo reference after hardware discovery" << std::endl;
    return false;
  }

  xnn_deinitialize();
  return true;
}
#endif

}  // namespace

int main() {
  if (!TestSequentialConsumers() || !TestConcurrentConsumers()) {
    return EXIT_FAILURE;
  }

#if defined(ORT_CPUINFO_TEST_USE_XNNPACK)
  if (!TestXnnpackReleasesCpuinfo()) {
    return EXIT_FAILURE;
  }
#endif

  return EXIT_SUCCESS;
}
