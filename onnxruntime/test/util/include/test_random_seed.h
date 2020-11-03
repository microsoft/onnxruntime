// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace test {

// These environment variables control the behavior of GetTestRandomSeed().
namespace test_random_seed_env_vars {
// Specifies a fixed seed value to return.
// If set, this has the highest precedence.
constexpr const char* kValue = "ORT_TEST_RANDOM_SEED_VALUE";
// If set to 1 (and not using a fixed value), specifies that a new seed value is returned each time.
// The default behavior is to return the same cached seed value per process.
// This is useful when repeatedly running flaky tests to reproduce errors.
constexpr const char* kDoNotCache = "ORT_TEST_RANDOM_SEED_DO_NOT_CACHE";
}  // namespace test_random_seed_env_vars

using RandomSeedType = uint32_t;

/**
 * Gets a test random seed value.
 */
RandomSeedType GetTestRandomSeed();

}  // namespace test
}  // namespace onnxruntime
