// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunTest(int64_t k,
                    const std::vector<float>& input_vals,
                    const std::vector<int64_t>& input_dimensions,
                    const std::vector<float>& expected_vals,
                    const std::vector<int64_t>& expected_indices,
                    const std::vector<int64_t>& expected_dimensions,
                    int64_t axis = -1,
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                    const std::string& expected_err_str = "") {
  OpTester test("TopK");
  test.AddAttribute("k", k);
  if (axis != -1) {
    test.AddAttribute("axis", axis);
  }

  test.AddInput<float>("X", input_dimensions, input_vals);
  test.AddOutput<float>("Values", expected_dimensions, expected_vals);
  test.AddOutput<int64_t>("Indices", expected_dimensions, expected_indices);
  test.Run(expect_result, expected_err_str);
}

TEST(TopKOperator, Top1DefaultAxis) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1};
  std::vector<int64_t> expected_dimensions = {2, 1};
  RunTest(1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions);
}

TEST(TopKOperator, Top2DefaultAxis) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 1};
  std::vector<int64_t> expected_dimensions = {2, 2};
  RunTest(2, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions);
}

TEST(TopKOperator, Top3DefaultAxis) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.2f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 2, 1, 3};
  std::vector<int64_t> expected_dimensions = {2, 3};
  RunTest(3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions);
}

TEST(TopKOperator, TopAllDefaultAxis) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 0, 1, 2, 3, 0};
  std::vector<int64_t> expected_dimensions = {2, 4};
  RunTest(4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions);
}

TEST(TopKOperator, Top1ExplicitAxis) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.3f, 0.4f};
  std::vector<int64_t> expected_indices = {3, 1};
  std::vector<int64_t> expected_dimensions = {1, 2};
  int64_t axis = 0;
  RunTest(1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, axis);
}
/*
TEST(TopKOperator, Top2ExplicitAxis) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 1};
  std::vector<int64_t> expected_dimensions = {2, 2};
  int64_t axis = 0;
  RunTest(2, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, axis);
}

TEST(TopKOperator, Top3ExplicitAxis) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.2f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 2, 1, 3};
  std::vector<int64_t> expected_dimensions = {3, 2};
  int64_t axis = 0;
  RunTest(3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, axis);
}

TEST(TopKOperator, TopAllExplicitAxis) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 0, 1, 2, 3, 0};
  std::vector<int64_t> expected_dimensions = {4, 2};
  int64_t axis = 0;
  RunTest(4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, axis);
}
*/
TEST(TopKOperator, InvalidK) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 0, 1, 2, 3, 0};
  std::vector<int64_t> expected_dimensions = {2, 4};
  RunTest(0,
          input_vals,
          input_dimensions,
          expected_vals,
          expected_indices,
          expected_dimensions,
          1,
          OpTester::ExpectResult::kExpectFailure,
          "Invalid value for attribute k");
}

}  // namespace test
}  // namespace onnxruntime
