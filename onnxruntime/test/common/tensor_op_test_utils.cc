// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"

#include "test/util/include/test_random_seed.h"

namespace onnxruntime {
namespace test {

RandomValueGenerator::RandomValueGenerator()
    : generator_{static_cast<decltype(generator_)::result_type>(GetTestRandomSeed())} {
}

}  // namespace test
}  // namespace onnxruntime
