// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "../test_utils.h"

namespace onnxruntime {
namespace test {
namespace telum {

/**
 * @brief Test activation functions
 */
class TelumActivationTest : public TelumTestBase {};

// ============================================================================
// Relu Tests
// ============================================================================

TEST_F(TelumActivationTest, Relu_Basic) {
  OpTester test("Relu", 13);

  std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  auto expected = ComputeReluReference(input);

  test.AddInput<float>("X", {5}, input);
  test.AddOutput<float>("Y", {5}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Relu_AllNegative) {
  OpTester test("Relu", 13);

  std::vector<float> input = {-5.0f, -4.0f, -3.0f, -2.0f, -1.0f};
  std::vector<float> expected(5, 0.0f);  // All zeros

  test.AddInput<float>("X", {5}, input);
  test.AddOutput<float>("Y", {5}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Relu_AllPositive) {
  OpTester test("Relu", 13);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto expected = input;  // Unchanged

  test.AddInput<float>("X", {5}, input);
  test.AddOutput<float>("Y", {5}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Relu_2D) {
  OpTester test("Relu", 13);

  std::vector<float> input = {
    -1.0f, 2.0f, -3.0f, 4.0f,
    5.0f, -6.0f, 7.0f, -8.0f
  };
  auto expected = ComputeReluReference(input);

  test.AddInput<float>("X", {2, 4}, input);
  test.AddOutput<float>("Y", {2, 4}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Relu_Large) {
  OpTester test("Relu", 13);

  auto input = GenerateRandomFloats(1024, -5.0f, 5.0f, 300);
  auto expected = ComputeReluReference(input);

  test.AddInput<float>("X", {32, 32}, input);
  test.AddOutput<float>("Y", {32, 32}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

// ============================================================================
// Gelu Tests
// ============================================================================

TEST_F(TelumActivationTest, Gelu_Basic) {
  OpTester test("Gelu", 1, kMSDomain);

  std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  auto expected = ComputeGeluReference(input);

  test.AddInput<float>("X", {5}, input);
  test.AddOutput<float>("Y", {5}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Gelu_ZeroInput) {
  OpTester test("Gelu", 1, kMSDomain);

  std::vector<float> input(10, 0.0f);
  auto expected = ComputeGeluReference(input);

  test.AddInput<float>("X", {10}, input);
  test.AddOutput<float>("Y", {10}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Gelu_2D) {
  OpTester test("Gelu", 1, kMSDomain);

  auto input = GenerateSequentialFloats(24, -2.0f, 0.2f);
  auto expected = ComputeGeluReference(input);

  test.AddInput<float>("X", {4, 6}, input);
  test.AddOutput<float>("Y", {4, 6}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Gelu_TransformerSize) {
  OpTester test("Gelu", 1, kMSDomain);

  // Typical transformer FFN intermediate size
  auto input = GenerateRandomFloats(512 * 3072, -3.0f, 3.0f, 400);
  auto expected = ComputeGeluReference(input);

  test.AddInput<float>("X", {512, 3072}, input);
  test.AddOutput<float>("Y", {512, 3072}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

// ============================================================================
// Tanh Tests
// ============================================================================

TEST_F(TelumActivationTest, Tanh_Basic) {
  OpTester test("Tanh", 13);

  std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  auto expected = ComputeTanhReference(input);

  test.AddInput<float>("X", {5}, input);
  test.AddOutput<float>("Y", {5}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Tanh_LargeValues) {
  OpTester test("Tanh", 13);

  std::vector<float> input = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
  auto expected = ComputeTanhReference(input);

  // Tanh saturates at ±1 for large values
  test.AddInput<float>("X", {5}, input);
  test.AddOutput<float>("Y", {5}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Tanh_2D) {
  OpTester test("Tanh", 13);

  auto input = GenerateRandomFloats(100, -3.0f, 3.0f, 500);
  auto expected = ComputeTanhReference(input);

  test.AddInput<float>("X", {10, 10}, input);
  test.AddOutput<float>("Y", {10, 10}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

// ============================================================================
// Sigmoid Tests
// ============================================================================

TEST_F(TelumActivationTest, Sigmoid_Basic) {
  OpTester test("Sigmoid", 13);

  std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  std::vector<float> expected(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    expected[i] = 1.0f / (1.0f + std::exp(-input[i]));
  }

  test.AddInput<float>("X", {5}, input);
  test.AddOutput<float>("Y", {5}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Sigmoid_Zero) {
  OpTester test("Sigmoid", 13);

  std::vector<float> input(10, 0.0f);
  std::vector<float> expected(10, 0.5f);  // sigmoid(0) = 0.5

  test.AddInput<float>("X", {10}, input);
  test.AddOutput<float>("Y", {10}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

// ============================================================================
// Exp Tests
// ============================================================================

TEST_F(TelumActivationTest, Exp_Basic) {
  OpTester test("Exp", 13);

  std::vector<float> input = {0.0f, 1.0f, 2.0f};
  std::vector<float> expected = {1.0f, std::exp(1.0f), std::exp(2.0f)};

  test.AddInput<float>("X", {3}, input);
  test.AddOutput<float>("Y", {3}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Exp_Negative) {
  OpTester test("Exp", 13);

  std::vector<float> input = {-2.0f, -1.0f, 0.0f};
  std::vector<float> expected(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    expected[i] = std::exp(input[i]);
  }

  test.AddInput<float>("X", {3}, input);
  test.AddOutput<float>("Y", {3}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

// ============================================================================
// Log Tests
// ============================================================================

TEST_F(TelumActivationTest, Log_Basic) {
  OpTester test("Log", 13);

  std::vector<float> input = {1.0f, 2.0f, std::exp(1.0f)};
  std::vector<float> expected = {0.0f, std::log(2.0f), 1.0f};

  test.AddInput<float>("X", {3}, input);
  test.AddOutput<float>("Y", {3}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

// ============================================================================
// Sqrt Tests
// ============================================================================

TEST_F(TelumActivationTest, Sqrt_Basic) {
  OpTester test("Sqrt", 13);

  std::vector<float> input = {0.0f, 1.0f, 4.0f, 9.0f, 16.0f};
  std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};

  test.AddInput<float>("X", {5}, input);
  test.AddOutput<float>("Y", {5}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

TEST_F(TelumActivationTest, Sqrt_2D) {
  OpTester test("Sqrt", 13);

  std::vector<float> input = {
    1.0f, 4.0f, 9.0f, 16.0f,
    25.0f, 36.0f, 49.0f, 64.0f
  };
  std::vector<float> expected(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    expected[i] = std::sqrt(input[i]);
  }

  test.AddInput<float>("X", {2, 4}, input);
  test.AddOutput<float>("Y", {2, 4}, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTelumExecutionProvider});
}

}  // namespace telum
}  // namespace test
}  // namespace onnxruntime

// Made with Bob
