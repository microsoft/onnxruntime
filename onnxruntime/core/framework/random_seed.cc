// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "random_seed.h"

#include <atomic>
#include <chrono>
#include <limits>

namespace onnxruntime {
namespace utils {

namespace {
// value indicating that the random seed should be generated
const uint64_t k_generate_random_seed_value{
    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1};

std::atomic<uint64_t> g_initial_random_seed{k_generate_random_seed_value};

uint32_t GenerateRandomSeedFromTime() {
  return static_cast<uint32_t>(
      std::chrono::system_clock::now().time_since_epoch().count());
}
}  // namespace

uint32_t GetStaticRandomSeed(uint32_t default_seed) {
  uint64_t init = g_initial_random_seed.load();
  static const uint32_t k_random_seed{
      [init](uint32_t default_seed) {
        // The priority is: seed set by SetStaticRandomSeed > positive default_seed > random seed from time.
        return init != k_generate_random_seed_value ? static_cast<uint32_t>(init) : (default_seed > 0 ? default_seed : GenerateRandomSeedFromTime());
      }(default_seed)};
  return k_random_seed;
}

void SetStaticRandomSeed(uint32_t seed) {
  g_initial_random_seed.store(seed);
}

}  // namespace test
}  // namespace onnxruntime
