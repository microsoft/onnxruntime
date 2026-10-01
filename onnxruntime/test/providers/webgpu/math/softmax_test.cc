// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
TEST(SoftmaxOperator, webgpu_nan) {
  OpTester test("Softmax", 13);  // axis default is -1

  std::vector<float> x_vals = {-INFINITY, -INFINITY, -INFINITY};
  std::vector<float> expected_result = {0.0f, 0.0f, 0.0f};
  std::vector<int64_t> dimensions = {1, 3};

  test.AddInput<float>("X", dimensions, x_vals);
  test.AddOutput<float>("Y", dimensions, expected_result);

  // explicitly disable for EPs that do not handle NaN
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCpuExecutionProvider, kCoreMLExecutionProvider, kDmlExecutionProvider});
}
#endif

}  // namespace test
}  // namespace onnxruntime
