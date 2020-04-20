// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"

namespace onnxruntime {
namespace test {

RandomValueGenerator::RandomValueGenerator()
    : random_seed_{GetTestRandomSeed()},
      generator_{static_cast<decltype(generator_)::result_type>(random_seed_)},
      output_trace_{__FILE__, __LINE__, "ORT test random seed: " + std::to_string(random_seed_)} {
}

}  // namespace test
}  // namespace onnxruntime
