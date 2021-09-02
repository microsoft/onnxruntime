// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Some Sparse specific coverage. Since we are
// re-using the axis and shape handling code, we
// rely on dense tests here.
TEST(SparseSqueezeOpTest, Squeeze_1) {
  // squeeze leading 1
  std::vector<int64_t> dense_shape{1, 8};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices{3, 5, 7};
  std::vector<int64_t> expected_shape{8};

  OpTester test("Squeeze", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, flat_indices);
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddSparseCooOutput("squeezed", expected_shape, values, flat_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SparseSqueezeOpTest, Squeeze_1_Two_D_Indices) {
  // squeeze leading 1
  std::vector<int64_t> dense_shape{1, 8};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> two_d_indices{0, 3, 0, 5, 0, 7};
  std::vector<int64_t> expected_shape{8};
  std::vector<int64_t> flat_indices{3, 5, 7};

  OpTester test("Squeeze", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, two_d_indices);
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddSparseCooOutput("squeezed", expected_shape, values, flat_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SparseSqueezeOpTest, Squeeze_Empty_Axes_1) {
  // No axes input, squeeze all ones
  std::vector<int64_t> dense_shape{1, 8};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices{3, 5, 7};
  std::vector<int64_t> expected_shape{8};

  OpTester test("Squeeze", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, flat_indices);
  test.AddSparseCooOutput("squeezed", expected_shape, values, flat_indices);
  // TensorRT doesn't seem to support missing 'axes'
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SparseSqueezeOpTest, Squeeze_Empty_Axes_2) {
  // nothing to "squeeze" out in the input shape
  std::vector<int64_t> dense_shape{4, 2};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices{3, 5, 7};
  std::vector<int64_t> expected_shape{4, 2};

  OpTester test("Squeeze", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, flat_indices);
  test.AddSparseCooOutput("squeezed", expected_shape, values, flat_indices);
  // TensorRT doesn't seem to support missing 'axes'
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SparseSqueezeOpTest, Squeeze_Empty_Axes_3) {
  std::vector<int64_t> dense_shape{1,1};
  std::vector<float> values{3.f};
  std::vector<int64_t> flat_indices{0};
  std::vector<int64_t> expected_shape{1};

  OpTester test("Squeeze", 1, onnxruntime::kMSDomain);
  // Squeeze all for all 1's shape will end up as a scalar, however
  // for sparse we enforce that the shape is not empty
  test.AddSparseCooInput("data", dense_shape, values, flat_indices);
  test.AddSparseCooOutput("squeezed", expected_shape, values, flat_indices);
  // TensorRT doesn't seem to support missing 'axes'
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SparseUnsqueeze, Unsqueeze_1) {
  std::vector<int64_t> dense_shape{2};
  std::vector<float> values{3.f};
  std::vector<int64_t> flat_indices{0};
  std::vector<int64_t> expected_shape{2, 1};

  OpTester test("Unsqueeze", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, flat_indices);
  test.AddInput<int64_t>("axes", {1}, {1});
  test.AddSparseCooOutput("unsqueezed", expected_shape, values, flat_indices);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SparseUnsqueeze, Unsqueeze_2) {
  std::vector<int64_t> dense_shape{2};
  std::vector<float> values{3.f};
  std::vector<int64_t> flat_indices{0};
  std::vector<int64_t> expected_shape{1, 2};

  OpTester test("Unsqueeze", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, flat_indices);
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddSparseCooOutput("unsqueezed", expected_shape, values, flat_indices);
  test.Run();
}


}
}  // namespace onnxruntime

#endif  //  !defined(DISABLE_SPARSE_TENSORS)
