// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "../test_utils.h"

namespace onnxruntime {
namespace test {
namespace telum {

/**
 * @brief Test Gemm (General Matrix Multiplication) kernel
 * Gemm: Y = alpha * A * B + beta * C
 */
class TelumGemmTest : public TelumTestBase {};

// ============================================================================
// Basic Gemm Tests
// ============================================================================

TEST_F(TelumGemmTest, Gemm_Basic) {
  OpTester test("Gemm", 13);

  // Y = A * B + C
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};  // 2x2
  std::vector<float> C = {1.0f, 1.0f, 1.0f, 1.0f};  // 2x2

  auto expected = ComputeGemmReference(A, B, C, 2, 2, 2, 1.0f, 1.0f);

  test.AddInput<float>("A", {2, 2}, A);
  test.AddInput<float>("B", {2, 2}, B);
  test.AddInput<float>("C", {2, 2}, C);
  test.AddOutput<float>("Y", {2, 2}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_NoBias) {
  OpTester test("Gemm", 13);

  // Y = A * B (no bias)
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};

  auto expected = ComputeGemmReference(A, B, {}, 2, 2, 2, 1.0f, 0.0f);

  test.AddInput<float>("A", {2, 2}, A);
  test.AddInput<float>("B", {2, 2}, B);
  test.AddOutput<float>("Y", {2, 2}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_WithAlpha) {
  OpTester test("Gemm", 13);
  test.AddAttribute("alpha", 2.0f);

  // Y = 2.0 * A * B + C
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {1.0f, 0.0f, 0.0f, 1.0f};  // Identity
  std::vector<float> C = {1.0f, 1.0f, 1.0f, 1.0f};

  auto expected = ComputeGemmReference(A, B, C, 2, 2, 2, 2.0f, 1.0f);

  test.AddInput<float>("A", {2, 2}, A);
  test.AddInput<float>("B", {2, 2}, B);
  test.AddInput<float>("C", {2, 2}, C);
  test.AddOutput<float>("Y", {2, 2}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_WithBeta) {
  OpTester test("Gemm", 13);
  test.AddAttribute("beta", 2.0f);

  // Y = A * B + 2.0 * C
  std::vector<float> A = {1.0f, 0.0f, 0.0f, 1.0f};  // Identity
  std::vector<float> B = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> C = {1.0f, 1.0f, 1.0f, 1.0f};

  auto expected = ComputeGemmReference(A, B, C, 2, 2, 2, 1.0f, 2.0f);

  test.AddInput<float>("A", {2, 2}, A);
  test.AddInput<float>("B", {2, 2}, B);
  test.AddInput<float>("C", {2, 2}, C);
  test.AddOutput<float>("Y", {2, 2}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_WithAlphaBeta) {
  OpTester test("Gemm", 13);
  test.AddAttribute("alpha", 0.5f);
  test.AddAttribute("beta", 2.0f);

  // Y = 0.5 * A * B + 2.0 * C
  std::vector<float> A = {2.0f, 4.0f, 6.0f, 8.0f};
  std::vector<float> B = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> C = {1.0f, 2.0f, 3.0f, 4.0f};

  auto expected = ComputeGemmReference(A, B, C, 2, 2, 2, 0.5f, 2.0f);

  test.AddInput<float>("A", {2, 2}, A);
  test.AddInput<float>("B", {2, 2}, B);
  test.AddInput<float>("C", {2, 2}, C);
  test.AddOutput<float>("Y", {2, 2}, expected);
  RunOnTelum(test);
}

// ============================================================================
// Non-Square Matrix Tests
// ============================================================================

TEST_F(TelumGemmTest, Gemm_RectangularMatrices) {
  OpTester test("Gemm", 13);

  // A: 3x2, B: 2x4, C: 3x4, Y: 3x4
  auto A = GenerateSequentialFloats(6, 1.0f, 1.0f);   // 3x2
  auto B = GenerateSequentialFloats(8, 1.0f, 1.0f);   // 2x4
  auto C = GenerateSequentialFloats(12, 0.5f, 0.0f);  // 3x4

  auto expected = ComputeGemmReference(A, B, C, 3, 2, 4, 1.0f, 1.0f);

  test.AddInput<float>("A", {3, 2}, A);
  test.AddInput<float>("B", {2, 4}, B);
  test.AddInput<float>("C", {3, 4}, C);
  test.AddOutput<float>("Y", {3, 4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_VectorMatrix) {
  OpTester test("Gemm", 13);

  // A: 1x4 (row vector), B: 4x3, C: 1x3, Y: 1x3
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  auto B = GenerateSequentialFloats(12, 1.0f, 1.0f);
  std::vector<float> C = {1.0f, 1.0f, 1.0f};

  auto expected = ComputeGemmReference(A, B, C, 1, 4, 3, 1.0f, 1.0f);

  test.AddInput<float>("A", {1, 4}, A);
  test.AddInput<float>("B", {4, 3}, B);
  test.AddInput<float>("C", {1, 3}, C);
  test.AddOutput<float>("Y", {1, 3}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_MatrixVector) {
  OpTester test("Gemm", 13);

  // A: 4x3, B: 3x1 (column vector), C: 4x1, Y: 4x1
  auto A = GenerateSequentialFloats(12, 1.0f, 1.0f);
  std::vector<float> B = {1.0f, 2.0f, 3.0f};
  std::vector<float> C = {0.5f, 0.5f, 0.5f, 0.5f};

  auto expected = ComputeGemmReference(A, B, C, 4, 3, 1, 1.0f, 1.0f);

  test.AddInput<float>("A", {4, 3}, A);
  test.AddInput<float>("B", {3, 1}, B);
  test.AddInput<float>("C", {4, 1}, C);
  test.AddOutput<float>("Y", {4, 1}, expected);
  RunOnTelum(test);
}

// ============================================================================
// Transformer-Sized Tests
// ============================================================================

TEST_F(TelumGemmTest, Gemm_BertHiddenProjection) {
  OpTester test("Gemm", 13);

  // Typical BERT hidden projection: [batch*seq, hidden] x [hidden, hidden]
  // Simplified: [64, 768] x [768, 768] + [64, 768]
  const int M = 64;
  const int K = 768;
  const int N = 768;

  auto A = GenerateRandomFloats(M * K, -0.1f, 0.1f, 100);
  auto B = GenerateRandomFloats(K * N, -0.1f, 0.1f, 200);
  auto C = GenerateRandomFloats(M * N, -0.01f, 0.01f, 300);

  auto expected = ComputeGemmReference(A, B, C, M, K, N, 1.0f, 1.0f);

  test.AddInput<float>("A", {M, K}, A);
  test.AddInput<float>("B", {K, N}, B);
  test.AddInput<float>("C", {M, N}, C);
  test.AddOutput<float>("Y", {M, N}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_BertFFNIntermediate) {
  OpTester test("Gemm", 13);

  // BERT FFN intermediate: [batch*seq, hidden] x [hidden, 4*hidden]
  // Simplified: [128, 768] x [768, 3072] + [128, 3072]
  const int M = 128;
  const int K = 768;
  const int N = 3072;

  auto A = GenerateRandomFloats(M * K, -0.1f, 0.1f, 400);
  auto B = GenerateRandomFloats(K * N, -0.05f, 0.05f, 500);
  auto C = GenerateRandomFloats(M * N, -0.01f, 0.01f, 600);

  auto expected = ComputeGemmReference(A, B, C, M, K, N, 1.0f, 1.0f);

  test.AddInput<float>("A", {M, K}, A);
  test.AddInput<float>("B", {K, N}, B);
  test.AddInput<float>("C", {M, N}, C);
  test.AddOutput<float>("Y", {M, N}, expected);
  RunOnTelum(test);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(TelumGemmTest, Gemm_ZeroAlpha) {
  OpTester test("Gemm", 13);
  test.AddAttribute("alpha", 0.0f);

  // Y = 0 * A * B + C = C
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> C = {9.0f, 10.0f, 11.0f, 12.0f};

  auto expected = C;  // Result should be C

  test.AddInput<float>("A", {2, 2}, A);
  test.AddInput<float>("B", {2, 2}, B);
  test.AddInput<float>("C", {2, 2}, C);
  test.AddOutput<float>("Y", {2, 2}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_ZeroBeta) {
  OpTester test("Gemm", 13);
  test.AddAttribute("beta", 0.0f);

  // Y = A * B + 0 * C = A * B
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> C = {100.0f, 100.0f, 100.0f, 100.0f};

  auto expected = ComputeGemmReference(A, B, {}, 2, 2, 2, 1.0f, 0.0f);

  test.AddInput<float>("A", {2, 2}, A);
  test.AddInput<float>("B", {2, 2}, B);
  test.AddInput<float>("C", {2, 2}, C);
  test.AddOutput<float>("Y", {2, 2}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_IdentityMatrix) {
  OpTester test("Gemm", 13);

  // A * I + C where I is identity
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {1.0f, 0.0f, 0.0f, 1.0f};  // Identity
  std::vector<float> C = {0.5f, 0.5f, 0.5f, 0.5f};

  auto expected = ComputeGemmReference(A, B, C, 2, 2, 2, 1.0f, 1.0f);

  test.AddInput<float>("A", {2, 2}, A);
  test.AddInput<float>("B", {2, 2}, B);
  test.AddInput<float>("C", {2, 2}, C);
  test.AddOutput<float>("Y", {2, 2}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_NegativeValues) {
  OpTester test("Gemm", 13);

  std::vector<float> A = {-1.0f, 2.0f, -3.0f, 4.0f};
  std::vector<float> B = {5.0f, -6.0f, 7.0f, -8.0f};
  std::vector<float> C = {-1.0f, -1.0f, -1.0f, -1.0f};

  auto expected = ComputeGemmReference(A, B, C, 2, 2, 2, 1.0f, 1.0f);

  test.AddInput<float>("A", {2, 2}, A);
  test.AddInput<float>("B", {2, 2}, B);
  test.AddInput<float>("C", {2, 2}, C);
  test.AddOutput<float>("Y", {2, 2}, expected);
  RunOnTelum(test);
}

TEST_F(TelumGemmTest, Gemm_BroadcastBias) {
  OpTester test("Gemm", 13);

  // C can be broadcast from [1, N] to [M, N]
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 3x2
  std::vector<float> B = {1.0f, 2.0f, 3.0f, 4.0f};              // 2x2
  std::vector<float> C = {1.0f, 1.0f};                          // 1x2 (broadcast to 3x2)

  auto expected = ComputeGemmReference(A, B, C, 3, 2, 2, 1.0f, 1.0f);

  test.AddInput<float>("A", {3, 2}, A);
  test.AddInput<float>("B", {2, 2}, B);
  test.AddInput<float>("C", {1, 2}, C);
  test.AddOutput<float>("Y", {3, 2}, expected);
  RunOnTelum(test);
}

}  // namespace telum
}  // namespace test
}  // namespace onnxruntime

// Made with Bob
