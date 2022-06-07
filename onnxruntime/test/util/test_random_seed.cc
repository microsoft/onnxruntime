// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/test_random_seed.h"

#include <chrono>

#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace test {

RandomSeedType GetTestRandomSeed() {
  static const auto fixed_random_seed =
      ParseEnvironmentVariable<RandomSeedType>(test_random_seed_env_vars::kValue);
  if (fixed_random_seed.has_value()) {
    // use fixed value
    return *fixed_random_seed;
  }

  auto generate_from_time = []() {
    return static_cast<RandomSeedType>(
        std::chrono::steady_clock::now().time_since_epoch().count());
  };

  static const auto use_cached =
      !ParseEnvironmentVariableWithDefault<bool>(test_random_seed_env_vars::kDoNotCache, false);
  if (use_cached) {
    // initially generate from current time
    static const auto static_random_seed = generate_from_time();
    return static_random_seed;
  }

  // generate from current time
  return generate_from_time();
}

}  // namespace test
}  // namespace onnxruntime
