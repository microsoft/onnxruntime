
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#ifdef USE_CUDA
TEST(IsFiniteTest, Float) {
  OpTester test("IsFinite", 9);

  std::vector<int64_t> shape = {3};
  std::vector<float> input = {std::numeric_limits<float>::infinity(),
                              1.0f, std::numeric_limits<float>::quiet_NaN()};

  test.AddInput<float>("X", shape, input);
  test.AddOutput<bool>("Y", shape, {false, true, false});

  test.Run();
}

TEST(IsFiniteTest, Double) {
  OpTester test("IsFinite", 9);

  std::vector<int64_t> shape = {3};
  std::vector<double> input = {std::numeric_limits<double>::infinity(),
                              1.0f, std::numeric_limits<double>::quiet_NaN()};

  test.AddInput<double>("X", shape, input);
  test.AddOutput<bool>("Y", shape, {false, true, false});

  test.Run();
}

TEST(IsFiniteTest, MLFloat16) {
  OpTester test("IsFinite", 9);

  std::vector<int64_t> shape = {3};
  std::vector<float> input = {std::numeric_limits<float>::infinity(),
                              1.0f, std::numeric_limits<float>::quiet_NaN()};
  std::vector<MLFloat16> input_half(input.size());
  ConvertFloatToMLFloat16(input.data(), input_half.data(), int(input.size()));

  test.AddInput<MLFloat16>("X", shape, input_half);
  test.AddOutput<bool>("Y", shape, {false, true, false});

  test.Run();
}
#endif

}  // namespace test
}  // namespace onnxruntime