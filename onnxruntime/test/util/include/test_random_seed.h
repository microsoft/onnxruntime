// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace test {

using RandomSeedType = uint32_t;

// Possible improvement:
// We could make this a bit nicer by setting the seed with a GTest
// ::testing::Environment and registering that as a global environment.
// That way we could get a different generated seed on each test run when using
// --gtest_repeat.
// That was the initial approach, but there were some issues with the Mac CI
// build in onnxruntime_shared_lib_test.

/**
 * Gets the test random seed value which does not change during the test run.
 * The random seed value is obtained as follows, in order:
 * 1. environment variable ORT_TEST_RANDOM_SEED, if available and valid
 * 2. generated from current time
 */
RandomSeedType GetTestRandomSeed();

inline const char* GetTestRandomSeedEnvironmentVariableName() {
  return "ORT_TEST_RANDOM_SEED";
}

}  // namespace test
}  // namespace onnxruntime
