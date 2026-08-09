// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/test_random_seed.h"

namespace onnxruntime {
namespace test {

FixedPatternValueGenerator::FixedPatternValueGenerator()
    : generator_{0},
      output_trace_{__FILE__, __LINE__, "ORT test random seed with fixed pattern tensor generator"} {
}

}  // namespace test
}  // namespace onnxruntime
