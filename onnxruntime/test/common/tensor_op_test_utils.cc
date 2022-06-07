// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/test_random_seed.h"

namespace onnxruntime {
namespace test {

RandomValueGenerator::RandomValueGenerator(optional<RandomSeedType> seed)
    : random_seed_{
          seed.has_value() ? *seed : static_cast<RandomSeedType>(GetTestRandomSeed())},
      generator_{random_seed_},
      output_trace_{__FILE__, __LINE__, "ORT test random seed: " + std::to_string(random_seed_)} {
}

}  // namespace test
}  // namespace onnxruntime
