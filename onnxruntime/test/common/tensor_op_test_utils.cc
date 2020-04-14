// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_seed.h"
#include "test/common/tensor_op_test_utils.h"

#include <chrono>

namespace onnxruntime {
namespace test {

RandomValueGenerator::RandomValueGenerator()
    : generator_{static_cast<decltype(generator_)::result_type>(utils::GetRandomSeed())} {
}

}  // namespace test
}  // namespace onnxruntime
