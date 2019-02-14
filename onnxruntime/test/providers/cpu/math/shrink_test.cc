// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

template <typename T>
static void RunTest(float bias,
					float lambd,
					const std::vector<T>& input_vals,
                    const std::vector<int64_t>& input_dimensions,
                    const std::vector<T>& expected_vals,
                    const std::vector<int64_t>& expected_dimensions,
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                    const std::string& expected_err_str = "") {
  OpTester test("Shrink", 9);
  if (bias != 0.0f) {
    test.AddAttribute("bias", bias);
  }
  if (lambd != 0.5f) {
    test.AddAttribute("lambd", lambd);
  }

  test.AddInput<float>("X", input_dimensions, input_vals);
  test.AddOutput<float>("Values", expected_dimensions, expected_vals);
  test.Run(expect_result, expected_err_str);
}

TEST(ShrinkOperator, FloatTypeDefaultBiasDefaultLambd) {
  std::vector<float> input_vals = {-1.0f, -0.4f, 0.4f, 1.0f};
  std::vector<int64_t> input_dimensions = {2, 2};
  std::vector<float> expected_vals = {-1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<int64_t> expected_dimensions = {2, 2};
  RunTest<float>(0.0f, 0.5f, input_vals, input_dimensions, expected_vals, expected_dimensions);
}

TEST(ShrinkOperator, FloatTypeNonDefaultBiasNonDefaultLambd) {
  std::vector<float> input_vals = {-1.0f, -0.4f, 0.4f, 1.0f};
  std::vector<int64_t> input_dimensions = {2, 2};
  std::vector<float> expected_vals = {9.0f, 0.0f, 0.0f, -9.0f};
  std::vector<int64_t> expected_dimensions = {2, 2};
  RunTest<float>(10.0f, 0.4f, input_vals, input_dimensions, expected_vals, expected_dimensions);
}

}  // namespace test
}  // namespace onnxruntime