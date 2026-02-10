// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "../test_utils.h"

namespace onnxruntime {
namespace test {
namespace telum {

/**
 * @brief Test MatMul kernel with various shapes and data
 */
class TelumMatMulTest : public TelumTestBase {};

TEST_F(TelumMatMulTest, BasicMatMul_2x3_3x4) {
  // Test: [2,3] × [3,4] = [2,4]
  OpTester test("MatMul", 13);

  // Input A: 2x3
  std::vector<float> A = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
  };

  // Input B: 3x4
  std::vector<float> B = {
    1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f,
    9.0f, 10.0f, 11.0f, 12.0f
  };

  // Expected output: 2x4
  auto expected = ComputeMatMulReference(A, B, 2, 3, 4);

  test.AddInput<float>("A", {2, 3}, A);
  test.AddInput<float>("B", {3, 4}, B);
  test.AddOutput<float>("Y", {2, 4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumMatMulTest, SquareMatMul_4x4) {
  // Test: [4,4] × [4,4] = [4,4]
  OpTester test("MatMul", 13);

  auto A = GenerateSequentialFloats(16, 1.0f, 1.0f);
  auto B = GenerateSequentialFloats(16, 1.0f, 0.5f);
  auto expected = ComputeMatMulReference(A, B, 4, 4, 4);

  test.AddInput<float>("A", {4, 4}, A);
  test.AddInput<float>("B", {4, 4}, B);
  test.AddOutput<float>("Y", {4, 4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumMatMulTest, VectorMatMul_1x10_10x1) {
  // Test: [1,10] × [10,1] = [1,1] (dot product)
  OpTester test("MatMul", 13);

  auto A = GenerateSequentialFloats(10, 1.0f, 1.0f);
  auto B = GenerateSequentialFloats(10, 1.0f, 1.0f);
  auto expected = ComputeMatMulReference(A, B, 1, 10, 1);

  test.AddInput<float>("A", {1, 10}, A);
  test.AddInput<float>("B", {10, 1}, B);
  test.AddOutput<float>("Y", {1, 1}, expected);
  RunOnTelum(test);
}

TEST_F(TelumMatMulTest, LargeMatMul_128x128) {
  // Test larger matrices typical in transformers
  OpTester test("MatMul", 13);

  constexpr int64_t M = 128, K = 128, N = 128;
  auto A = GenerateRandomFloats(M * K, -1.0f, 1.0f, 42);
  auto B = GenerateRandomFloats(K * N, -1.0f, 1.0f, 43);
  auto expected = ComputeMatMulReference(A, B, M, K, N);

  test.AddInput<float>("A", {M, K}, A);
  test.AddInput<float>("B", {K, N}, B);
  test.AddOutput<float>("Y", {M, N}, expected);
  RunOnTelum(test);
}

TEST_F(TelumMatMulTest, TransformerSizes_512x768) {
  // Test typical transformer hidden size
  OpTester test("MatMul", 13);

  constexpr int64_t M = 512, K = 768, N = 768;  // BERT-base dimensions
  auto A = GenerateRandomFloats(M * K, -0.5f, 0.5f, 100);
  auto B = GenerateRandomFloats(K * N, -0.5f, 0.5f, 101);
  auto expected = ComputeMatMulReference(A, B, M, K, N);

  test.AddInput<float>("A", {M, K}, A);
  test.AddInput<float>("B", {K, N}, B);
  test.AddOutput<float>("Y", {M, N}, expected);
  RunOnTelum(test);
}

TEST_F(TelumMatMulTest, BatchedMatMul_2x3x4_2x4x5) {
  // Test batched matrix multiplication
  OpTester test("MatMul", 13);

  // Batch of 2, each [3,4] × [4,5] = [3,5]
  auto A = GenerateSequentialFloats(2 * 3 * 4, 1.0f, 1.0f);
  auto B = GenerateSequentialFloats(2 * 4 * 5, 1.0f, 0.5f);

  // Compute expected for each batch
  std::vector<float> expected;
  for (int b = 0; b < 2; ++b) {
    std::vector<float> A_batch(A.begin() + b * 12, A.begin() + (b + 1) * 12);
    std::vector<float> B_batch(B.begin() + b * 20, B.begin() + (b + 1) * 20);
    auto batch_result = ComputeMatMulReference(A_batch, B_batch, 3, 4, 5);
    expected.insert(expected.end(), batch_result.begin(), batch_result.end());
  }

  test.AddInput<float>("A", {2, 3, 4}, A);
  test.AddInput<float>("B", {2, 4, 5}, B);
  test.AddOutput<float>("Y", {2, 3, 5}, expected);
  RunOnTelum(test);
}

TEST_F(TelumMatMulTest, ZeroMatrix) {
  // Test with zero matrix
  OpTester test("MatMul", 13);

  std::vector<float> A(12, 0.0f);  // 3x4 zeros
  auto B = GenerateSequentialFloats(20, 1.0f, 1.0f);  // 4x5
  std::vector<float> expected(15, 0.0f);  // 3x5 zeros

  test.AddInput<float>("A", {3, 4}, A);
  test.AddInput<float>("B", {4, 5}, B);
  test.AddOutput<float>("Y", {3, 5}, expected);
  RunOnTelum(test);
}

TEST_F(TelumMatMulTest, IdentityMatrix) {
  // Test with identity matrix
  OpTester test("MatMul", 13);

  // Identity matrix 4x4
  std::vector<float> I = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  };

  auto A = GenerateSequentialFloats(16, 1.0f, 1.0f);

  // A × I = A
  test.AddInput<float>("A", {4, 4}, A);
  test.AddInput<float>("B", {4, 4}, I);
  test.AddOutput<float>("Y", {4, 4}, A);
  RunOnTelum(test);
}

TEST_F(TelumMatMulTest, NegativeValues) {
  // Test with negative values
  OpTester test("MatMul", 13);

  std::vector<float> A = {
    -1.0f, -2.0f, -3.0f,
    -4.0f, -5.0f, -6.0f
  };

  std::vector<float> B = {
    1.0f, 2.0f,
    3.0f, 4.0f,
    5.0f, 6.0f
  };

  auto expected = ComputeMatMulReference(A, B, 2, 3, 2);

  test.AddInput<float>("A", {2, 3}, A);
  test.AddInput<float>("B", {3, 2}, B);
  test.AddOutput<float>("Y", {2, 2}, expected);
  RunOnTelum(test);
}

TEST_F(TelumMatMulTest, MixedSignValues) {
  // Test with mixed positive and negative values
  OpTester test("MatMul", 13);

  auto A = GenerateRandomFloats(20, -2.0f, 2.0f, 200);
  auto B = GenerateRandomFloats(30, -2.0f, 2.0f, 201);
  auto expected = ComputeMatMulReference(A, B, 4, 5, 6);

  test.AddInput<float>("A", {4, 5}, A);
  test.AddInput<float>("B", {5, 6}, B);
  test.AddOutput<float>("Y", {4, 6}, expected);
  RunOnTelum(test);
}

}  // namespace telum
}  // namespace test
}  // namespace onnxruntime

// Made with Bob
