// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunTest(int op_set,
                    int64_t k,
                    const std::vector<float>& input_vals,
                    const std::vector<int64_t>& input_dimensions,
                    const std::vector<float>& expected_vals,
                    const std::vector<int64_t>& expected_indices,
                    const std::vector<int64_t>& expected_dimensions,
                    bool is_tensorrt_supported = true,
                    int64_t axis = -1,
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                    const std::string& expected_err_str = "") {
  OpTester test("TopK", op_set);

  // Attributes
  if (axis != -1)
    test.AddAttribute("axis", axis);
  if (op_set <= 9)
    test.AddAttribute("k", k);

  // Inputs
  test.AddInput<float>("X", input_dimensions, input_vals);
  if (op_set == 10)
    test.AddInput<int64_t>("K", {1}, {k});

  // Outputs
  test.AddOutput<float>("Values", expected_dimensions, expected_vals);
  test.AddOutput<int64_t>("Indices", expected_dimensions, expected_indices);

  // Run test and check results
  std::unordered_set<std::string> excluded_providers;
  if (!is_tensorrt_supported) {
    excluded_providers.insert(kTensorrtExecutionProvider);//Disable TensorRT because of unsupported data types
  }
  test.Run(expect_result, expected_err_str, excluded_providers);
}

TEST(TopKOperator, Top1DefaultAxisOpset9) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1};
  std::vector<int64_t> expected_dimensions = {2, 1};
  RunTest(9, 1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, Top2DefaultAxisOpset9) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 1};
  std::vector<int64_t> expected_dimensions = {2, 2};
  RunTest(9, 2, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, Top3DefaultAxisOpset9) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.2f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 2, 1, 3};
  std::vector<int64_t> expected_dimensions = {2, 3};
  RunTest(9, 3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, TopAllDefaultAxisOpset9) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 0, 1, 2, 3, 0};
  std::vector<int64_t> expected_dimensions = {2, 4};
  RunTest(9, 4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, Top1ExplicitAxisOpset9) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.3f, 0.4f};
  std::vector<int64_t> expected_indices = {3, 1};
  std::vector<int64_t> expected_dimensions = {1, 2};
  int64_t axis = 0;
  RunTest(9, 1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, Top2ExplicitAxisOpset9) {
  std::vector<float> input_vals = {0.0f, 1.0f, 2.0f, 11.0f, 08.0f, 5.0f, 6.0f, 7.0f, 4.0f, 9.0f, 10.0f, 3.0f};
  std::vector<int64_t> input_dimensions = {3, 4};
  std::vector<float> expected_vals = {8.0f, 9.0f, 10.0f, 11.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<int64_t> expected_indices = {1, 2, 2, 0, 2, 1, 1, 1};
  std::vector<int64_t> expected_dimensions = {2, 4};
  int64_t axis = 0;
  RunTest(9, 2, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, Top3ExplicitAxisOpset9) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.3f, 0.4f, 0.2f, 0.3f, 0.1f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1, 1, 0, 0, 2};
  std::vector<int64_t> expected_dimensions = {3, 2};
  int64_t axis = 0;
  RunTest(9, 3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, TopAllExplicitAxisOpset9) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.3f, 0.4f, 0.2f, 0.3f, 0.1f, 0.3f, 0.1f, 0.2f};
  std::vector<int64_t> expected_indices = {3, 1, 1, 0, 0, 2, 2, 3};
  std::vector<int64_t> expected_dimensions = {4, 2};
  int64_t axis = 0;
  RunTest(9, 4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, TopAllExplicitAxis1DInputOpset9) {
  std::vector<float> input_vals = {93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f, 285.0f, 527.0f, 862.0f};
  std::vector<int64_t> input_dimensions = {13};
  std::vector<float> expected_vals = {983.0f, 978.0f, 971.0f, 862.0f, 723.0f, 695.0f, 531.0f, 527.0f, 483.0f, 285.0f, 247.0f, 242.0f, 93.0f};
  std::vector<int64_t> expected_indices = {7, 3, 2, 12, 9, 1, 8, 11, 4, 10, 5, 6, 0};
  std::vector<int64_t> expected_dimensions = {13};
  int64_t axis = 0;
  RunTest(9, 13, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, Top1ExplicitAxisMultiDInputOpset9) {
  std::vector<float> input_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<int64_t> input_dimensions = {2, 2, 2};
  std::vector<float> expected_vals = {3, 4, 7, 8};
  std::vector<int64_t> expected_indices = {1, 1, 1, 1};
  std::vector<int64_t> expected_dimensions = {2, 1, 2};
  int64_t axis = 1;
  RunTest(9, 1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, InvalidKOpset9) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 0, 1, 2, 3, 0};
  std::vector<int64_t> expected_dimensions = {2, 4};
  RunTest(9,
          0,
          input_vals,
          input_dimensions,
          expected_vals,
          expected_indices,
          expected_dimensions,
          true,
          1,
          OpTester::ExpectResult::kExpectFailure,
          "Invalid value for attribute k");
}

TEST(TopKOperator, Top1DefaultAxisOpset10) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1};
  std::vector<int64_t> expected_dimensions = {2, 1};
  RunTest(10, 1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, Top2DefaultAxisOpset10) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 1};
  std::vector<int64_t> expected_dimensions = {2, 2};
  RunTest(10, 2, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, Top3DefaultAxisOpset10) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.2f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 2, 1, 3};
  std::vector<int64_t> expected_dimensions = {2, 3};
  RunTest(10, 3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, TopAllDefaultAxisOpset10) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 0, 1, 2, 3, 0};
  std::vector<int64_t> expected_dimensions = {2, 4};
  RunTest(10, 4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, Top1ExplicitAxisOpset10) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.3f, 0.4f};
  std::vector<int64_t> expected_indices = {3, 1};
  std::vector<int64_t> expected_dimensions = {1, 2};
  int64_t axis = 0;
  RunTest(10, 1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, Top2ExplicitAxisOpset10) {
  std::vector<float> input_vals = {0.0f, 1.0f, 2.0f, 11.0f, 08.0f, 5.0f, 6.0f, 7.0f, 4.0f, 9.0f, 10.0f, 3.0f};
  std::vector<int64_t> input_dimensions = {3, 4};
  std::vector<float> expected_vals = {8.0f, 9.0f, 10.0f, 11.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<int64_t> expected_indices = {1, 2, 2, 0, 2, 1, 1, 1};
  std::vector<int64_t> expected_dimensions = {2, 4};
  int64_t axis = 0;
  RunTest(10, 2, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, Top3ExplicitAxisOpset10) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.3f, 0.4f, 0.2f, 0.3f, 0.1f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1, 1, 0, 0, 2};
  std::vector<int64_t> expected_dimensions = {3, 2};
  int64_t axis = 0;
  RunTest(10, 3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, TopAllExplicitAxisOpset10) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.3f, 0.4f, 0.2f, 0.3f, 0.1f, 0.3f, 0.1f, 0.2f};
  std::vector<int64_t> expected_indices = {3, 1, 1, 0, 0, 2, 2, 3};
  std::vector<int64_t> expected_dimensions = {4, 2};
  int64_t axis = 0;
  RunTest(10, 4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, TopAllExplicitAxis1DInputOpset10) {
  std::vector<float> input_vals = {93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f, 285.0f, 527.0f, 862.0f};
  std::vector<int64_t> input_dimensions = {13};
  std::vector<float> expected_vals = {983.0f, 978.0f, 971.0f, 862.0f, 723.0f, 695.0f, 531.0f, 527.0f, 483.0f, 285.0f, 247.0f, 242.0f, 93.0f};
  std::vector<int64_t> expected_indices = {7, 3, 2, 12, 9, 1, 8, 11, 4, 10, 5, 6, 0};
  std::vector<int64_t> expected_dimensions = {13};
  int64_t axis = 0;
  RunTest(10, 13, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, Top1ExplicitAxisMultiDInputOpset10) {
  std::vector<float> input_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<int64_t> input_dimensions = {2, 2, 2};
  std::vector<float> expected_vals = {3, 4, 7, 8};
  std::vector<int64_t> expected_indices = {1, 1, 1, 1};
  std::vector<int64_t> expected_dimensions = {2, 1, 2};
  int64_t axis = 1;
  RunTest(10, 1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);
}

TEST(TopKOperator, InvalidKOpset10) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 0, 1, 2, 3, 0};
  std::vector<int64_t> expected_dimensions = {2, 4};
  RunTest(10,
          0,
          input_vals,
          input_dimensions,
          expected_vals,
          expected_indices,
          expected_dimensions,
          false,
          1,
          OpTester::ExpectResult::kExpectFailure,
          "value of k should be greater than 0");
}

}  // namespace test
}  // namespace onnxruntime
