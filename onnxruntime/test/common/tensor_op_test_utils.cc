// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_seed.h"
#include "test/common/tensor_op_test_utils.h"

#include <chrono>

namespace onnxruntime {
namespace test {

static uint32_t GetSeedValue(RandomValueGenerator::RandomSeedType random_seed_type) {
  switch (random_seed_type) {
    case RandomValueGenerator::RandomSeedType::kStatic:
      return 42;
    case RandomValueGenerator::RandomSeedType::kPerProcess:
      return utils::GetStaticRandomSeed();
    default:  // dynamic
      return static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  }
}

RandomValueGenerator::RandomValueGenerator(RandomSeedType random_seed_type)
    : generator_{static_cast<decltype(generator_)::result_type>(GetSeedValue(random_seed_type))} {
}

}  // namespace test
}  // namespace onnxruntime
