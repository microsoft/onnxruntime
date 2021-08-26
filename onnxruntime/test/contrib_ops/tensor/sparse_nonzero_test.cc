// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/sparse_utils.h"

namespace onnxruntime {
namespace test {

// Funny tests on SparseTensors
TEST(SparseNonZero, TestAllZeros2D) {
  std::vector<int64_t> dense_shape{4, 2};
  std::vector<float> values{0.f, 0.f, 0.f};
  std::vector<int64_t> flat_indices{3, 5, 7};
  std::vector<int64_t> two_d_indieces{1, 1, 2, 1, 3, 1};
  const std::vector<int64_t> expected_output_shape{2, 0};
  const std::vector<int64_t> expected_output;
  {
    // Try flat indices first
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, flat_indices);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  {
    // 2-D indices
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, two_d_indieces);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseNonZero, TestFullySparse2D) {
  std::vector<int64_t> dense_shape{4, 2};
  std::vector<float> values;
  std::vector<int64_t> flat_indices;
  std::vector<int64_t> two_d_indieces;
  const std::vector<int64_t> expected_output_shape{2, 0};
  const std::vector<int64_t> expected_output;
  {
    // Try flat indices first
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, flat_indices);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  {
    // 2-D indices
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, two_d_indieces);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseNonZero, TestAllNonZeros2D) {
  // When the dense shape is 2-D, indices can be 1 or 2-D
  std::vector<int64_t> dense_shape{4,2};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices{3, 5, 7};
  std::vector<int64_t> two_d_indieces{1,1,2,1,3,1};
  const std::vector<int64_t> expected_output_shape{2,3};
  const std::vector<int64_t> expected_output{1,2,3,1,1,1};

  {
    // Try flat indices first
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, flat_indices);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  {
    // 2-D indices
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, two_d_indieces);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseNonZero, TestSomeNonZeros2D) {
  // When the dense shape is 2-D, indices can be 1 or 2-D
  std::vector<int64_t> dense_shape{4, 2};
  std::vector<float> values{3.f, 0.f, 5.f, 0.f, 7.f};
  std::vector<int64_t> flat_indices{3, 4, 5, 5, 7};
  std::vector<int64_t> two_d_indieces{1, 1, 2, 0, 2, 1, 3, 0, 3, 1};
  const std::vector<int64_t> expected_output_shape{2, 3};
  const std::vector<int64_t> expected_output{1, 2, 3, 1, 1, 1};

  {
    // Try flat indices first
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, flat_indices);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  {
    // 2-D indices
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, two_d_indieces);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseNonZero, TestAllZeros1D) {
  std::vector<int64_t> dense_shape{8};
  std::vector<float> values{0.f, 0.f, 0.f};
  std::vector<int64_t> flat_indices{3, 5, 7};
  const std::vector<int64_t> expected_output_shape{1, 0};
  const std::vector<int64_t> expected_output;

  {
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, flat_indices);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseNonZero, TestSomeZeros1D) {
  std::vector<int64_t> dense_shape{8};
  std::vector<float> values{3.f, 0.f, 5.f, 0.f, 7.f};
  std::vector<int64_t> flat_indices{3, 4, 5, 6, 7};
  const std::vector<int64_t> expected_output_shape{1, 3};
  const std::vector<int64_t> expected_output{3, 5, 7};

  {
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, flat_indices);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseNonZero, TestAllNonZeros1D) {
  std::vector<int64_t> dense_shape{8};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices{3, 5, 7};
  const std::vector<int64_t> expected_output_shape{1, 3};
  const std::vector<int64_t> expected_output{3,5,7};

  {
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, flat_indices);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(SparseNonZero, TestFullySparse1D) {
  std::vector<int64_t> dense_shape{8};
  std::vector<float> values;
  std::vector<int64_t> flat_indices;
  const std::vector<int64_t> expected_output_shape{1, 0};
  const std::vector<int64_t> expected_output;

  {
    OpTester tester("NonZero", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", dense_shape, values, flat_indices);
    tester.AddOutput("Y", expected_output_shape, expected_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

}
}  // namespace onnxruntime

#endif  //  !defined(DISABLE_SPARSE_TENSORS)
