// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// Verify that MaxPoolGrad rejects indices that exceed the output buffer size.
TEST(MaxPoolGradTest, IndicesOutOfRange) {
  OpTester test("MaxPoolGrad", 9, kOnnxDomain);

  // dY: shape [1, 1, 2, 2]
  test.AddInput<float>("dY", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
  // Indices: same shape, last value 100 is out of range [0, 9)
  test.AddInput<int64_t>("Indices", {1, 1, 2, 2}, {0, 1, 2, 100});
  // Expected dX: shape [1, 1, 3, 3] → 9 elements
  test.AddOutput<float>("dX", {1, 1, 3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Invalid index in MaxPoolGrad: index value 100 is out of range [0, 9).",
           {}, nullptr, &execution_providers);
}

// Verify that MaxPoolGrad rejects negative indices.
TEST(MaxPoolGradTest, IndicesNegative) {
  OpTester test("MaxPoolGrad", 9, kOnnxDomain);

  // dY: shape [1, 1, 2, 2]
  test.AddInput<float>("dY", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
  // Indices: same shape, value -1 is negative and out of range
  test.AddInput<int64_t>("Indices", {1, 1, 2, 2}, {0, 1, -1, 3});
  // Expected dX: shape [1, 1, 3, 3] → 9 elements
  test.AddOutput<float>("dX", {1, 1, 3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Invalid index in MaxPoolGrad: index value -1 is out of range [0, 9).",
           {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
