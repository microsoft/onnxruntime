// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "random_seed.h"
#include "random_generator.h"

#include <atomic>
#include <chrono>

namespace onnxruntime {
namespace utils {

static std::atomic<int64_t> g_random_seed(std::chrono::system_clock::now().time_since_epoch().count());

int64_t GetRandomSeed() {
  return g_random_seed.load();
}

void SetRandomSeed(int64_t seed) {
  g_random_seed.store(seed);

  // Reset default generators.
  RandomGenerator::Default().SetSeed(seed);
  PhiloxGenerator::Default().SetSeed(static_cast<uint64_t>(seed));
}

}  // namespace utils
}  // namespace onnxruntime
