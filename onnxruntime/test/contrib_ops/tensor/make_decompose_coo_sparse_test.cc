// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/sparse_utils.h"

namespace onnxruntime {
namespace test {

TEST(SparseMakeCoo, TestFlatIndices) {
  const std::vector<int64_t> dense_shape{4, 2};
  const std::vector<float> values{3.f, 5.f, 7.f};
  const std::vector<int64_t> flat_indices{3, 5, 7};

  OpTester tester("MakeCooSparse", 1, onnxruntime::kMSDomain);
  tester.AddInput("DenseShape", {2}, dense_shape);
  tester.AddInput("Values", {3}, values);
  tester.AddInput("Indices", {3}, flat_indices);
  tester.AddSparseCooOutput("Output", dense_shape, gsl::make_span(values), flat_indices);
  tester.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(SparseMakeCoo, Test2DIndices) {
  const std::vector<int64_t> dense_shape{4, 2};
  const std::vector<float> values{3.f, 5.f, 7.f};
  const std::vector<int64_t> indices{1, 1, 2, 1, 3, 1};

  OpTester tester("MakeCooSparse", 1, onnxruntime::kMSDomain);
  tester.AddInput("DenseShape", {2}, dense_shape);
  tester.AddInput("Values", {3}, values);
  tester.AddInput("Indices", {3, 2}, indices);
  tester.AddSparseCooOutput("Output", dense_shape, gsl::make_span(values), indices);
  tester.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(SparseMakeCoo, TestFullySparse) {
  const std::vector<int64_t> dense_shape{4, 2};
  const std::vector<float> values;
  const std::vector<int64_t> indices;

  OpTester tester("MakeCooSparse", 1, onnxruntime::kMSDomain);
  tester.AddInput("DenseShape", {2}, dense_shape);
  tester.AddInput("Values", {0}, values);
  tester.AddInput("Indices", {0}, indices);
  tester.AddSparseCooOutput("Output", dense_shape, gsl::make_span(values), indices);
  tester.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(SparseDecomposeCoo, TestFlatIndices) {
  const std::vector<int64_t> dense_shape{4, 2};
  const std::vector<float> values{3.f, 5.f, 7.f};
  const std::vector<int64_t> flat_indices{3, 5, 7};

  OpTester tester("SparseDecomposeToDense", 1, onnxruntime::kMSDomain);
  tester.AddSparseCooInput("SparseCooInput", dense_shape, values, flat_indices);
  tester.AddOutput("DenseShape", {2}, dense_shape);
  tester.AddOutput("Values", {3}, values);
  tester.AddOutput("Indices", {3}, flat_indices);
  tester.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(SparseDecomposeCoo, Test2DIndices) {
  const std::vector<int64_t> dense_shape{4, 2};
  const std::vector<float> values{3.f, 5.f, 7.f};
  const std::vector<int64_t> indices{1, 1, 2, 1, 3, 1};
  const std::vector<int64_t> expected_indices{3, 5, 7};

  OpTester tester("SparseDecomposeToDense", 1, onnxruntime::kMSDomain);
  tester.AddSparseCooInput("SparseCooInput", dense_shape, values, indices);
  tester.AddOutput("DenseShape", {2}, dense_shape);
  tester.AddOutput("Values", {3}, values);
  tester.AddOutput("Indices", {3}, expected_indices);
  tester.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(SparseDecomposeCoo, TestFullySparse) {
  const std::vector<int64_t> dense_shape{4, 2};
  const std::vector<float> values;
  const std::vector<int64_t> indices;

  OpTester tester("SparseDecomposeToDense", 1, onnxruntime::kMSDomain);
  tester.AddSparseCooInput("SparseCooInput", dense_shape, values, indices);
  tester.AddOutput("DenseShape", {2}, dense_shape);
  tester.AddOutput("Values", {0}, values);
  tester.AddOutput("Indices", {0}, indices);
  tester.Run(OpTester::ExpectResult::kExpectSuccess);
}



}
}  // namespace onnxruntime

#endif  //  !defined(DISABLE_SPARSE_TENSORS)