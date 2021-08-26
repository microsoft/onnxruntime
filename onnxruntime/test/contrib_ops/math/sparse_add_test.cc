// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/sparse_utils.h"

namespace onnxruntime {
namespace test {

TEST(SparseAdd, TestFullySparse2D) {
  std::vector<int64_t> dense_shape_1{4, 2};
  std::vector<int64_t> dense_shape_2{3, 2};
  std::vector<float> values;
  std::vector<int64_t> flat_indices;
  const std::vector<int64_t> expected_output_shape{4, 2};
  const std::vector<float> expected_output;
  {
    // Try flat indices first
    OpTester tester("Add", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape_1, values, flat_indices);
    tester.AddSparseCooInput("B", dense_shape_2, values, flat_indices);
    tester.AddSparseCooOutput("Y", expected_output_shape, expected_output, flat_indices);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseAdd, TestFullySparseA) {
  std::vector<int64_t> dense_shape_a{4, 2};
  std::vector<int64_t> dense_shape_b{3, 2};
  std::vector<float> values_a;
  std::vector<int64_t> flat_indices_a;
  std::vector<float> values_b{2.f, 4.f, 5.f};
  std::vector<int64_t> flat_indices_b{2, 4, 5};
  const std::vector<int64_t> expected_output_shape{4, 2};
  const std::vector<float> expected_output;
  {
    // Try flat indices first
    OpTester tester("Add", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape_a, values_a, flat_indices_a);
    tester.AddSparseCooInput("B", dense_shape_b, values_b, flat_indices_b);
    tester.AddSparseCooOutput("Y", expected_output_shape, values_b, flat_indices_b);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseAdd, TestFullySparseB) {
  std::vector<int64_t> dense_shape_a{4, 2};
  std::vector<int64_t> dense_shape_b{3, 2};
  std::vector<float> values_a{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices_a{3, 5, 7};
  std::vector<float> values_b;
  std::vector<int64_t> flat_indices_b;
  const std::vector<int64_t> expected_output_shape{4, 2};
  const std::vector<float> expected_output;
  {
    // Try flat indices first
    OpTester tester("Add", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape_a, values_a, flat_indices_a);
    tester.AddSparseCooInput("B", dense_shape_b, values_b, flat_indices_b);
    tester.AddSparseCooOutput("Y", expected_output_shape, values_a, flat_indices_a);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseAdd, TestAddFlatIndices) {
  std::vector<int64_t> dense_shape_a{4, 2};
  std::vector<int64_t> dense_shape_b{3, 2};
  std::vector<float> values_a{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices_a{3, 5, 7};
  std::vector<float> values_b{2.f, 4.f, 5.f};
  std::vector<int64_t> flat_indices_b{2, 4, 5};
  const std::vector<int64_t> expected_output_shape{4, 2};
  const std::vector<float> expected_output{2.f, 3.f, 4.f, 10.f, 7.f};
  const std::vector<int64_t> expected_indices{2, 3, 4, 5, 7};
  {
    // Try flat indices first
    OpTester tester("Add", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape_a, values_a, flat_indices_a);
    tester.AddSparseCooInput("B", dense_shape_b, values_b, flat_indices_b);
    tester.AddSparseCooOutput("Y", expected_output_shape, expected_output, expected_indices);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseAdd, TestAddTwoDIndices) {
  std::vector<int64_t> dense_shape_a{4, 2};
  std::vector<int64_t> dense_shape_b{3, 2};
  std::vector<float> values_a{3.f, 5.f, 7.f};
  std::vector<int64_t> TwoD_indices_a{1, 1, 2, 1, 3, 1};

  std::vector<float> values_b{2.f, 4.f, 5.f};
  std::vector<int64_t> TwoD_indices_b{1, 0, 2, 0, 2, 1};

  const std::vector<int64_t> expected_output_shape{4, 2};
  const std::vector<float> expected_output{2.f, 3.f, 4.f, 10.f, 7.f};
  const std::vector<int64_t> expected_indices{2, 3, 4, 5, 7};
  {
    // Try flat indices first
    OpTester tester("Add", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape_a, values_a, TwoD_indices_a);
    tester.AddSparseCooInput("B", dense_shape_b, values_b, TwoD_indices_b);
    tester.AddSparseCooOutput("Y", expected_output_shape, expected_output, expected_indices);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseAdd, TestMixedDimensions) {
  std::vector<int64_t> dense_shape_a{8};
  std::vector<int64_t> dense_shape_b{3, 2};

  std::vector<float> values_a{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices_a{3, 5, 7};

  std::vector<float> values_b{2.f, 4.f, 5.f};
  std::vector<int64_t> TwoD_indices_b{1, 0, 2, 0, 2, 1};

  const std::vector<int64_t> expected_output_shape{3, 8};
  const std::vector<float> expected_output{2.f, 3.f, 4.f, 10.f, 7.f};
  const std::vector<int64_t> expected_indices{2, 3, 4, 5, 7};
  {
    // Try flat indices first
    OpTester tester("Add", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape_a, values_a, flat_indices_a);
    tester.AddSparseCooInput("B", dense_shape_b, values_b, TwoD_indices_b);
    tester.AddSparseCooOutput("Y", expected_output_shape, expected_output, expected_indices);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}
}
}  // namespace onnxruntime

#endif  //  !defined(DISABLE_SPARSE_TENSORS)
