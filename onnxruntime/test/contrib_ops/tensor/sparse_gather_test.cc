// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/sparse_utils.h"

namespace onnxruntime {
namespace test {

TEST(SparseGatherTest, EmptyInput) {
  std::vector<int64_t> dense_shape{0};
  std::vector<float> values;
  std::vector<int64_t> flat_indices;
  std::vector<int64_t> gather_indices_shape{3};
  std::vector<int64_t> gather_indices{0, 1, 3};

  std::vector<int64_t> expected_shape{0};

  OpTester test("Gather", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, flat_indices);
  test.AddInput<int64_t>("indices", gather_indices_shape, gather_indices);
  test.AddSparseCooOutput("output", expected_shape, values, flat_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SparseGatherTest, EmptyIndices) {
  std::vector<int64_t> dense_shape{1, 8};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices{3, 5, 7};

  std::vector<int64_t> gather_indices_shape{0};
  std::vector<int64_t> gather_indices;

  std::vector<int64_t> expected_shape{1, 8};

  OpTester test("Gather", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, flat_indices);
  test.AddInput<int64_t>("indices", gather_indices_shape, gather_indices);
  test.AddSparseCooOutput("output", expected_shape, values, flat_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SparseGatherTest, NoMatch) {
  std::vector<int64_t> dense_shape{1, 8};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices{3, 5, 7};

  std::vector<int64_t> gather_indices_shape{3};
  std::vector<int64_t> gather_indices{0, 1, 6};

  std::vector<int64_t> expected_shape{1, 8};

  OpTester test("Gather", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, flat_indices);
  test.AddInput<int64_t>("indices", gather_indices_shape, gather_indices);
  test.AddSparseCooOutput<float>("output", expected_shape, {}, {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SparseGatherTest, SomeMatch) {
  std::vector<int64_t> dense_shape{1, 8};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices{3, 5, 7};

  std::vector<int64_t> gather_indices_shape{3};
  std::vector<int64_t> gather_indices{3, 5, 6};

  std::vector<float> expected_values{3.f, 5.f};
  std::vector<int64_t> expected_indices{3, 5};
  std::vector<int64_t> expected_shape{1, 8};

  OpTester test("Gather", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, flat_indices);
  test.AddInput<int64_t>("indices", gather_indices_shape, gather_indices);
  test.AddSparseCooOutput<float>("output", expected_shape, expected_values, expected_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SparseGatherTest, Two_D_Match) {
  std::vector<int64_t> dense_shape{1, 8};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> two_d_indices{0, 3, 0, 5, 0, 7};

  std::vector<int64_t> gather_indices_shape{3};
  std::vector<int64_t> gather_indices{3, 5, 6};

  std::vector<float> expected_values{3.f, 5.f};
  std::vector<int64_t> expected_indices{3, 5};
  std::vector<int64_t> expected_shape{1, 8};

  OpTester test("Gather", 1, onnxruntime::kMSDomain);
  test.AddSparseCooInput("data", dense_shape, values, two_d_indices);
  test.AddInput<int64_t>("indices", gather_indices_shape, gather_indices);
  test.AddSparseCooOutput<float>("output", expected_shape, expected_values, expected_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

}
}  // namespace onnxruntime


#endif  // DISABLE_SPARSE_TENSORS
