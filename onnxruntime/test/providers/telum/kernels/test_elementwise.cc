// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "../test_utils.h"

namespace onnxruntime {
namespace test {
namespace telum {

/**
 * @brief Test elementwise binary operations
 */
class TelumElementwiseTest : public TelumTestBase {};

// ============================================================================
// Add Tests
// ============================================================================

TEST_F(TelumElementwiseTest, Add_Basic) {
  OpTester test("Add", 13);

  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
  auto expected = ComputeAddReference(A, B);

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Add_2D) {
  OpTester test("Add", 13);

  auto A = GenerateSequentialFloats(12, 1.0f, 1.0f);
  auto B = GenerateSequentialFloats(12, 0.5f, 0.5f);
  auto expected = ComputeAddReference(A, B);

  test.AddInput<float>("A", {3, 4}, A);
  test.AddInput<float>("B", {3, 4}, B);
  test.AddOutput<float>("C", {3, 4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Add_Broadcast_Vector) {
  OpTester test("Add", 13);

  // A: [2, 4], B: [4] -> broadcast to [2, 4]
  const std::vector<float> A = {
      1.0f, 2.0f, 3.0f, 4.0f,
      -1.0f, -2.0f, -3.0f, -4.0f,
  };
  const std::vector<float> B = {10.0f, 20.0f, 30.0f, 40.0f};

  std::vector<float> expected(A.size());
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      expected[i * 4 + j] = A[i * 4 + j] + B[j];
    }
  }

  test.AddInput<float>("A", {2, 4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {2, 4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Add_Broadcast_Scalar) {
  OpTester test("Add", 13);

  // A: [2, 3, 4], B: [1] -> broadcast scalar
  const std::vector<float> A = GenerateSequentialFloats(2 * 3 * 4, 1.0f, 1.0f);
  const std::vector<float> B = {0.5f};

  std::vector<float> expected(A.size());
  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = A[i] + B[0];
  }

  test.AddInput<float>("A", {2, 3, 4}, A);
  test.AddInput<float>("B", {1}, B);
  test.AddOutput<float>("C", {2, 3, 4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Add_Negative) {
  OpTester test("Add", 13);

  std::vector<float> A = {-1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> B = {1.0f, 2.0f, 3.0f, 4.0f};
  auto expected = ComputeAddReference(A, B);

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Add_Large) {
  OpTester test("Add", 13);

  auto A = GenerateRandomFloats(1024, -10.0f, 10.0f, 100);
  auto B = GenerateRandomFloats(1024, -10.0f, 10.0f, 200);
  auto expected = ComputeAddReference(A, B);

  test.AddInput<float>("A", {32, 32}, A);
  test.AddInput<float>("B", {32, 32}, B);
  test.AddOutput<float>("C", {32, 32}, expected);
  RunOnTelum(test);
}

// ============================================================================
// Sub Tests
// ============================================================================

TEST_F(TelumElementwiseTest, Sub_Basic) {
  OpTester test("Sub", 13);

  std::vector<float> A = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> B = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected(A.size());

  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = A[i] - B[i];
  }

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Sub_Negative) {
  OpTester test("Sub", 13);

  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> expected(A.size());

  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = A[i] - B[i];
  }

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

// ============================================================================
// Mul Tests
// ============================================================================

TEST_F(TelumElementwiseTest, Mul_Basic) {
  OpTester test("Mul", 13);

  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> expected(A.size());

  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = A[i] * B[i];
  }

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Mul_Zero) {
  OpTester test("Mul", 13);

  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> expected(4, 0.0f);

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Mul_Negative) {
  OpTester test("Mul", 13);

  std::vector<float> A = {-1.0f, 2.0f, -3.0f, 4.0f};
  std::vector<float> B = {2.0f, -3.0f, 4.0f, -5.0f};
  std::vector<float> expected(A.size());

  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = A[i] * B[i];
  }

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

// ============================================================================
// Div Tests
// ============================================================================

TEST_F(TelumElementwiseTest, Div_Basic) {
  OpTester test("Div", 13);

  std::vector<float> A = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> B = {2.0f, 4.0f, 5.0f, 8.0f};
  std::vector<float> expected(A.size());

  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = A[i] / B[i];
  }

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Div_Fractional) {
  OpTester test("Div", 13);

  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {3.0f, 3.0f, 3.0f, 3.0f};
  std::vector<float> expected(A.size());

  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = A[i] / B[i];
  }

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

// ============================================================================
// Min Tests
// ============================================================================

TEST_F(TelumElementwiseTest, Min_Basic) {
  OpTester test("Min", 13);

  std::vector<float> A = {1.0f, 5.0f, 3.0f, 7.0f};
  std::vector<float> B = {2.0f, 4.0f, 6.0f, 1.0f};
  std::vector<float> expected(A.size());

  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = std::min(A[i], B[i]);
  }

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Min_Negative) {
  OpTester test("Min", 13);

  std::vector<float> A = {-1.0f, 2.0f, -3.0f, 4.0f};
  std::vector<float> B = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> expected(A.size());

  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = std::min(A[i], B[i]);
  }

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

// ============================================================================
// Max Tests
// ============================================================================

TEST_F(TelumElementwiseTest, Max_Basic) {
  OpTester test("Max", 13);

  std::vector<float> A = {1.0f, 5.0f, 3.0f, 7.0f};
  std::vector<float> B = {2.0f, 4.0f, 6.0f, 1.0f};
  std::vector<float> expected(A.size());

  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = std::max(A[i], B[i]);
  }

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

TEST_F(TelumElementwiseTest, Max_Negative) {
  OpTester test("Max", 13);

  std::vector<float> A = {-1.0f, 2.0f, -3.0f, 4.0f};
  std::vector<float> B = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> expected(A.size());

  for (size_t i = 0; i < A.size(); ++i) {
    expected[i] = std::max(A[i], B[i]);
  }

  test.AddInput<float>("A", {4}, A);
  test.AddInput<float>("B", {4}, B);
  test.AddOutput<float>("C", {4}, expected);
  RunOnTelum(test);
}

// ============================================================================
// 2D Tests for All Operations
// ============================================================================

TEST_F(TelumElementwiseTest, AllOps_2D) {
  // Test all operations with 2D tensors
  auto A = GenerateSequentialFloats(20, 1.0f, 1.0f);
  auto B = GenerateSequentialFloats(20, 0.5f, 0.5f);

  // Add
  {
    OpTester test("Add", 13);
    auto expected = ComputeAddReference(A, B);
    test.AddInput<float>("A", {4, 5}, A);
    test.AddInput<float>("B", {4, 5}, B);
    test.AddOutput<float>("C", {4, 5}, expected);
  RunOnTelum(test);
  }

  // Sub
  {
    OpTester test("Sub", 13);
    std::vector<float> expected(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
      expected[i] = A[i] - B[i];
    }
    test.AddInput<float>("A", {4, 5}, A);
    test.AddInput<float>("B", {4, 5}, B);
    test.AddOutput<float>("C", {4, 5}, expected);
  RunOnTelum(test);
  }

  // Mul
  {
    OpTester test("Mul", 13);
    std::vector<float> expected(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
      expected[i] = A[i] * B[i];
    }
    test.AddInput<float>("A", {4, 5}, A);
    test.AddInput<float>("B", {4, 5}, B);
    test.AddOutput<float>("C", {4, 5}, expected);
  RunOnTelum(test);
  }

  // Div
  {
    OpTester test("Div", 13);
    std::vector<float> expected(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
      expected[i] = A[i] / B[i];
    }
    test.AddInput<float>("A", {4, 5}, A);
    test.AddInput<float>("B", {4, 5}, B);
    test.AddOutput<float>("C", {4, 5}, expected);
  RunOnTelum(test);
  }

  // Min
  {
    OpTester test("Min", 13);
    std::vector<float> expected(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
      expected[i] = std::min(A[i], B[i]);
    }
    test.AddInput<float>("A", {4, 5}, A);
    test.AddInput<float>("B", {4, 5}, B);
    test.AddOutput<float>("C", {4, 5}, expected);
  RunOnTelum(test);
  }

  // Max
  {
    OpTester test("Max", 13);
    std::vector<float> expected(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
      expected[i] = std::max(A[i], B[i]);
    }
    test.AddInput<float>("A", {4, 5}, A);
    test.AddInput<float>("B", {4, 5}, B);
    test.AddOutput<float>("C", {4, 5}, expected);
  RunOnTelum(test);
  }
}

// ============================================================================
// Large Tensor Tests
// ============================================================================

TEST_F(TelumElementwiseTest, AllOps_Large) {
  const size_t size = 512 * 768;  // Transformer hidden size
  auto A = GenerateRandomFloats(size, -1.0f, 1.0f, 100);
  auto B = GenerateRandomFloats(size, -1.0f, 1.0f, 200);

  // Add
  {
    OpTester test("Add", 13);
    auto expected = ComputeAddReference(A, B);
    test.AddInput<float>("A", {512, 768}, A);
    test.AddInput<float>("B", {512, 768}, B);
    test.AddOutput<float>("C", {512, 768}, expected);
  RunOnTelum(test);
  }

  // Mul (commonly used for attention masks)
  {
    OpTester test("Mul", 13);
    std::vector<float> expected(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
      expected[i] = A[i] * B[i];
    }
    test.AddInput<float>("A", {512, 768}, A);
    test.AddInput<float>("B", {512, 768}, B);
    test.AddOutput<float>("C", {512, 768}, expected);
  RunOnTelum(test);
  }
}

}  // namespace telum
}  // namespace test
}  // namespace onnxruntime

// Made with Bob
