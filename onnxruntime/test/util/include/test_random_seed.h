// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gtest/gtest.h"

#include "core/common/optional.h"

namespace onnxruntime {
namespace test {

using RandomSeedType = uint32_t;

/**
 * Helper class to set the test random seed.
 * Pass an instance to gtest via ::testing::AddGlobalTestEnvironment().
 * The random seed value is obtained as follows, in order:
 * 1. initial_seed provided in constructor, if available
 * 2. environment variable ORT_TEST_RANDOM_SEED, if available and valid
 * 3. generated from current time
 */
class TestRandomSeedSetterEnvironment : public ::testing::Environment {
 public:
  /**
   * Constructor.
   * @param initial_seed If provided, this value will be set as the test
   *   random seed.
   */
  TestRandomSeedSetterEnvironment(
      optional<RandomSeedType> initial_seed = optional<RandomSeedType>{})
      : initial_seed_{std::move(initial_seed)} {}

 private:
  virtual void SetUp() override;

  const optional<RandomSeedType> initial_seed_;
};

RandomSeedType GetTestRandomSeed();

inline const char* GetTestRandomSeedEnvironmentVariableName() {
  return "ORT_TEST_RANDOM_SEED";
}

}  // namespace test
}  // namespace onnxruntime
