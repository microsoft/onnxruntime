// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(SparseOneHotEncoderTest, Integers) {
  const std::vector<int64_t> categories{0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<int64_t> input_shape{3};
  const std::vector<int64_t> input{2,4,6};

  const std::vector<int64_t> expected_dense_shape{3, 8};
  const std::vector<float> expected_values{1.0f, 1.0f, 1.0f};
  const std::vector<int64_t> expected_indices{2, 12, 22};

  OpTester test("OneHotEncoder", 1, onnxruntime::kMSDomain);
  test.AddAttribute("cats_int64s", categories);
  test.AddInput("X", input_shape, input);
  test.AddSparseCooOutput("Y", expected_dense_shape, expected_values, expected_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(SparseOneHotEncoderTest, ScalarInput) {
  const std::vector<int64_t> categories{0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<int64_t> input_shape;
  const std::vector<int64_t> input{4};

  const std::vector<int64_t> expected_dense_shape{8};
  const std::vector<float> expected_values{1.0f};
  const std::vector<int64_t> expected_indices{4};

  OpTester test("OneHotEncoder", 1, onnxruntime::kMSDomain);
  test.AddAttribute("cats_int64s", categories);
  test.AddInput("X", input_shape, input);
  test.AddSparseCooOutput("Y", expected_dense_shape, expected_values, expected_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}
}  // namespace onnxruntime

#endif // DISABLE_SPARSE_TENSORS
