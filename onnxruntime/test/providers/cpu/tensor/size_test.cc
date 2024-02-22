// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
void TestSizeOp(const std::initializer_list<int64_t>& dims) {
  OpTester test("Size");

  int64_t actual_size = 1;
  for (auto dim : dims)
    actual_size *= dim;

  std::vector<int64_t> dim_vector(dims);
  std::vector<T> input(actual_size);
  test.AddInput<T>("A", dim_vector, input);

  std::vector<int64_t> scalar_dims;
  test.AddOutput<int64_t>("B", scalar_dims, {actual_size});

  // Run tests. Disable TensorRT EP because of dynamic shape or unsupported data types
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Single-dimensional float tensor
TEST(SizeOpTest, Float1000Test) {
  TestSizeOp<float>({1000});
}

// Two-dimensional float tensor
TEST(SizeOpTest, Float3x3Test) {
  TestSizeOp<float>({3, 3});
}

// Three-dimensional float tensor
TEST(SizeOpTest, Float3x3x4Test) {
  TestSizeOp<float>({3, 3, 4});
}

// Int tensor
TEST(SizeOpTest, Int3x3Test) {
  TestSizeOp<int>({3, 3});
}

// Int scalar
TEST(SizeOpTest, IntScalarTest) {
  TestSizeOp<int>({});
}

}  // namespace test
}  // namespace onnxruntime
