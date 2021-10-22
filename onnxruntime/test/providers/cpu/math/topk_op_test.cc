// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T = float>
static void RunTest(int op_set,
                    int64_t k,
                    const std::vector<T>& input_vals,
                    const std::vector<int64_t>& input_dimensions,
                    const std::vector<T>& expected_vals,
                    const std::vector<int64_t>& expected_indices,
                    const std::vector<int64_t>& expected_dimensions,
                    bool is_tensorrt_supported = true,
                    int64_t axis = -1,
                    int64_t largest = 1,
                    int64_t sorted = 1,
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                    const std::string& expected_err_str = "") {
  OpTester test("TopK", op_set);

  // Attributes
  if (axis != -1)
    test.AddAttribute("axis", axis);
  if (op_set <= 9)
    test.AddAttribute("k", k);
  if (op_set == 11 && largest != 1)
    test.AddAttribute("largest", largest);
  if (op_set == 11 && sorted != 1)
    test.AddAttribute("sorted", sorted);

  // Inputs
  test.AddInput<T>("X", input_dimensions, input_vals);
  if (op_set >= 10)
    test.AddInput<int64_t>("K", {1}, {k});

  // Outputs
  if (sorted == 1) {
    test.AddOutput<T>("Values", expected_dimensions, expected_vals);
    test.AddOutput<int64_t>("Indices", expected_dimensions, expected_indices);
  } else {
    test.AddOutput<T>("Values", expected_dimensions, expected_vals, true);
    test.AddOutput<int64_t>("Indices", expected_dimensions, expected_indices, true);
  }

  // Run test and check results
  std::unordered_set<std::string> excluded_providers;
  if (!is_tensorrt_supported) {
    excluded_providers.insert(kTensorrtExecutionProvider);  //Disable TensorRT because of unsupported data types
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

TEST(TopKOperator, Top1DefaultAxisOpset9_double) {
  std::vector<double> input_vals = {0.1, 0.3, 0.2, 0.4, 0.1, 0.3, 0.3, 0.2};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<double> expected_vals = {0.4, 0.3};
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

TEST(TopKOperator, Top2DefaultAxisOpset9_double) {
  std::vector<double> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<double> expected_vals = {0.4f, 0.3f, 0.4f, 0.3f};
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

TEST(TopKOperator, Top3DefaultAxisOpset9_double) {
  std::vector<double> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<double> expected_vals = {0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.2f};
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

TEST(TopKOperator, TopAllDefaultAxisOpset9_double) {
  std::vector<double> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<double> expected_vals = {0.4f, 0.3f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f};
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

TEST(TopKOperator, Top1ExplicitAxisOpset9_double) {
  std::vector<double> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<double> expected_vals = {0.3f, 0.4f};
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

TEST(TopKOperator, Top2ExplicitAxisOpset9_double) {
  std::vector<double> input_vals = {0.0f, 1.0f, 2.0f, 11.0f, 08.0f, 5.0f, 6.0f, 7.0f, 4.0f, 9.0f, 10.0f, 3.0f};
  std::vector<int64_t> input_dimensions = {3, 4};
  std::vector<double> expected_vals = {8.0f, 9.0f, 10.0f, 11.0f, 4.0f, 5.0f, 6.0f, 7.0f};
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

TEST(TopKOperator, Top3ExplicitAxisOpset9_double) {
  std::vector<double> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<double> expected_vals = {0.3f, 0.4f, 0.2f, 0.3f, 0.1f, 0.3f};
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

TEST(TopKOperator, TopAllExplicitAxisOpset9_double) {
  std::vector<double> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<double> expected_vals = {0.3f, 0.4f, 0.2f, 0.3f, 0.1f, 0.3f, 0.1f, 0.2f};
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

TEST(TopKOperator, TopAllExplicitAxis1DInputOpset9_double) {
  std::vector<double> input_vals = {93.0, 695.0, 971.0, 978.0, 483.0, 247.0, 242.0, 983.0, 531.0, 723.0, 285.0, 527.0, 862.0};
  std::vector<int64_t> input_dimensions = {13};
  std::vector<double> expected_vals = {983.0, 978.0, 971.0, 862.0, 723.0, 695.0, 531.0, 527.0, 483.0, 285.0, 247.0, 242.0, 93.0};
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

TEST(TopKOperator, Top1ExplicitAxisMultiDInputOpset9_double) {
  std::vector<double> input_vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  std::vector<int64_t> input_dimensions = {2, 2, 2};
  std::vector<double> expected_vals = {3, 4, 7, 8};
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
          1,
          1,
          OpTester::ExpectResult::kExpectFailure,
          "Invalid value for attribute k");
}

TEST(TopKOperator, InvalidKOpset9_double) {
  std::vector<double> input_vals = {0.1, 0.3, 0.2, 0.4, 0.1, 0.3, 0.3, 0.2};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<double> expected_vals = {0.4, 0.3, 0.2, 0.1, 0.3, 0.3, 0.2, 0.1};
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
          1,
          1,
          OpTester::ExpectResult::kExpectFailure,
          "Invalid value for attribute k");
}

template <typename T>
static void top_0_default_axis(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<T> expected_vals = {};
  std::vector<int64_t> expected_indices = {};
  std::vector<int64_t> expected_dimensions = {2, 0};
  RunTest(opset_version, 0, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, -1, 1, sorted);
}

TEST(TopKOperator, Top0DefaultAxisLargestElements) {
  top_0_default_axis<float>(10);
  top_0_default_axis<float>(11);
  top_0_default_axis<float>(11, 0);  // unsorted
  top_0_default_axis<double>(10);
  top_0_default_axis<double>(11);
  top_0_default_axis<double>(11, 0);  // unsorted
}

template <typename T>
static void top_1_default_axis(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<T> expected_vals = {0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1};
  std::vector<int64_t> expected_dimensions = {2, 1};
  int64_t axis = -1;
  RunTest(opset_version, 1, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top1DefaultAxisLargestElements) {
  top_1_default_axis<float>(10);
  top_1_default_axis<float>(11);
  top_1_default_axis<float>(11, 0);  // unsorted
  top_1_default_axis<double>(10);
  top_1_default_axis<double>(11);
  top_1_default_axis<double>(11, 0);  // unsorted
}

template <typename T>
static void top_2_default_axis(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<T> expected_vals = {0.4f, 0.3f, 0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 1};
  std::vector<int64_t> expected_dimensions = {2, 2};
  int64_t axis = -1;
  RunTest(opset_version, 2, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top2DefaultAxisLargestElements) {
  top_2_default_axis<float>(10);
  top_2_default_axis<float>(11);
  top_2_default_axis<float>(11, 0);  // unsorted
  top_2_default_axis<double>(10);
  top_2_default_axis<double>(11);
  top_2_default_axis<double>(11, 0);  // unsorted
}

template <typename T>
static void top_3_default_axis(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<T> expected_vals = {0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.2f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 2, 1, 3};
  std::vector<int64_t> expected_dimensions = {2, 3};
  int64_t axis = -1;
  RunTest(opset_version, 3, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top3DefaultAxisLargestElements) {
  top_3_default_axis<float>(10);
  top_3_default_axis<float>(11);
  top_3_default_axis<float>(11, 0);  //unsorted
  top_3_default_axis<double>(10);
  top_3_default_axis<double>(11);
  top_3_default_axis<double>(11, 0);  //unsorted
}

template <typename T>
static void top_all_default_axis(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<T> expected_vals = {0.4f, 0.3f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 0, 1, 2, 3, 0};
  std::vector<int64_t> expected_dimensions = {2, 4};
  int64_t axis = -1;
  RunTest(opset_version, 4, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, TopAllDefaultAxisLargestElements) {
  top_all_default_axis<float>(10);
  top_all_default_axis<float>(11);
  top_all_default_axis<float>(11, 0);  // unsorted
  top_all_default_axis<double>(10);
  top_all_default_axis<double>(11);
  top_all_default_axis<double>(11, 0);  // unsorted
}

template <typename T>
static void top_1_explicit_axis(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<T> expected_vals = {0.3f, 0.4f};
  std::vector<int64_t> expected_indices = {3, 1};
  std::vector<int64_t> expected_dimensions = {1, 2};
  int64_t axis = 0;
  RunTest(opset_version, 1, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top1ExplicitAxisLargestElements) {
  top_1_explicit_axis<float>(10);
  top_1_explicit_axis<float>(11);
  top_1_explicit_axis<float>(11, 0);  // unsorted
  top_1_explicit_axis<double>(10);
  top_1_explicit_axis<double>(11);
  top_1_explicit_axis<double>(11, 0);  // unsorted
}

template <typename T>
static void top_2_explicit_axis(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.0f, 1.0f, 2.0f, 11.0f, 08.0f, 5.0f, 6.0f, 7.0f, 4.0f, 9.0f, 10.0f, 3.0f};
  std::vector<int64_t> input_dimensions = {3, 4};
  std::vector<T> expected_vals = {8.0f, 9.0f, 10.0f, 11.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<int64_t> expected_indices = {1, 2, 2, 0, 2, 1, 1, 1};
  std::vector<int64_t> expected_dimensions = {2, 4};
  int64_t axis = 0;
  RunTest(opset_version, 2, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top2ExplicitAxisLargestElements) {
  top_2_explicit_axis<float>(10);
  top_2_explicit_axis<float>(11);
  top_2_explicit_axis<float>(11, 0);  //unsorted
  top_2_explicit_axis<double>(10);
  top_2_explicit_axis<double>(11);
  top_2_explicit_axis<double>(11, 0);  //unsorted
}

template <typename T>
static void top_3_explicit_axis(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<T> expected_vals = {0.3f, 0.4f, 0.2f, 0.3f, 0.1f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1, 1, 0, 0, 2};
  std::vector<int64_t> expected_dimensions = {3, 2};
  int64_t axis = 0;
  RunTest(opset_version, 3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top3ExplicitAxisLargestElements) {
  top_3_explicit_axis<float>(10);
  top_3_explicit_axis<float>(11);
  top_3_explicit_axis<float>(11, 0);  //unsorted
  top_3_explicit_axis<double>(10);
  top_3_explicit_axis<double>(11);
  top_3_explicit_axis<double>(11, 0);  //unsorted
}

template <typename T>
static void top_all_explicit_axis(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<T> expected_vals = {0.3f, 0.4f, 0.2f, 0.3f, 0.1f, 0.3f, 0.1f, 0.2f};
  std::vector<int64_t> expected_indices = {3, 1, 1, 0, 0, 2, 2, 3};
  std::vector<int64_t> expected_dimensions = {4, 2};
  int64_t axis = 0;
  RunTest(opset_version, 4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, TopAllExplicitAxisLargestElements) {
  top_all_explicit_axis<float>(10);
  top_all_explicit_axis<float>(11);
  top_all_explicit_axis<float>(11, 0);  // unsorted
  top_all_explicit_axis<double>(10);
  top_all_explicit_axis<double>(11);
  top_all_explicit_axis<double>(11, 0);  // unsorted
}

template <typename T>
static void top_all_explicit_axis_1D_input(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f, 285.0f, 527.0f, 862.0f};
  std::vector<int64_t> input_dimensions = {13};
  std::vector<T> expected_vals = {983.0f, 978.0f, 971.0f, 862.0f, 723.0f, 695.0f, 531.0f, 527.0f, 483.0f, 285.0f, 247.0f, 242.0f, 93.0f};
  std::vector<int64_t> expected_indices = {7, 3, 2, 12, 9, 1, 8, 11, 4, 10, 5, 6, 0};
  std::vector<int64_t> expected_dimensions = {13};
  int64_t axis = 0;
  RunTest(opset_version, 13, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, TopAllExplicitAxis1DInputLargestElements) {
  top_all_explicit_axis_1D_input<float>(10);
  top_all_explicit_axis_1D_input<float>(11);
  top_all_explicit_axis_1D_input<float>(11, 0);  // unsorted
  top_all_explicit_axis_1D_input<double>(10);
  top_all_explicit_axis_1D_input<double>(11);
  top_all_explicit_axis_1D_input<double>(11, 0);  // unsorted
}

template <typename T>
static void top_2_explicit_axis_1D_large_input(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f,
                               93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f,
                               93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f,
                               93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f,
                               93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f,
                               93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f,
                               93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f,
                               93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f,
                               93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f,
                               93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f};

  std::vector<int64_t> input_dimensions = {100};
  std::vector<T> expected_vals = {983.0f, 983.0f};
  std::vector<int64_t> expected_indices = {7, 17};
  std::vector<int64_t> expected_dimensions = {2};
  int64_t axis = 0;
  RunTest(opset_version, 2, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, TopAllExplicitAxis1DLargeInputLargestElements) {
  top_2_explicit_axis_1D_large_input<float>(10);
  top_2_explicit_axis_1D_large_input<float>(11);
  top_2_explicit_axis_1D_large_input<float>(11, 0);  // unsorted
  top_2_explicit_axis_1D_large_input<double>(10);
  top_2_explicit_axis_1D_large_input<double>(11);
  top_2_explicit_axis_1D_large_input<double>(11, 0);  // unsorted
}

template <typename T>
static void top_1_explicit_axis_MultiD_input(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<int64_t> input_dimensions = {2, 2, 2};
  std::vector<T> expected_vals = {3, 4, 7, 8};
  std::vector<int64_t> expected_indices = {1, 1, 1, 1};
  std::vector<int64_t> expected_dimensions = {2, 1, 2};
  int64_t axis = 1;
  RunTest(opset_version, 1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top1ExplicitAxisMultiDInputLargestElements) {
  top_1_explicit_axis_MultiD_input<float>(10);
  top_1_explicit_axis_MultiD_input<float>(11);
  top_1_explicit_axis_MultiD_input<float>(11, 0);  // unsorted
  top_1_explicit_axis_MultiD_input<double>(10);
  top_1_explicit_axis_MultiD_input<double>(11);
  top_1_explicit_axis_MultiD_input<double>(11, 0);  // unsorted
}

template <typename T>
static void top_2_default_axis_smallest(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<T> expected_vals = {0.1f, 0.2f, 0.1f, 0.2f};
  std::vector<int64_t> expected_indices = {0, 2, 0, 3};
  std::vector<int64_t> expected_dimensions = {2, 2};
  int64_t axis = -1;
  RunTest(opset_version, 2, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 0, sorted);
}

TEST(TopKOperator, Top2DefaultAxisSmallestElements) {
  top_2_default_axis_smallest<float>(11);
  top_2_default_axis_smallest<float>(11, 0);  // unsorted
  top_2_default_axis_smallest<double>(11);
  top_2_default_axis_smallest<double>(11, 0);  // unsorted
}

template <typename T>
static void top_3_explicit_axis_smallest(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<T> expected_vals = {0.1f, 0.2f, 0.1f, 0.3f, 0.2f, 0.3f};
  std::vector<int64_t> expected_indices = {0, 3, 2, 0, 1, 2};
  std::vector<int64_t> expected_dimensions = {3, 2};
  int64_t axis = 0;
  RunTest(opset_version, 3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 0, sorted);
}

TEST(TopKOperator, Top3ExplicitAxisSmallestElements) {
  top_3_explicit_axis_smallest<float>(11);
  top_3_explicit_axis_smallest<float>(11, 0);  //unsorted
  top_3_explicit_axis_smallest<double>(11);
  top_3_explicit_axis_smallest<double>(11, 0);  //unsorted
}

template <typename T>
static void top_1_explicit_axis_MultiD_input_smallest(int opset_version, int64_t sorted = 1) {
  std::vector<T> input_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> input_dimensions = {2, 2, 2};
  std::vector<T> expected_vals = {1, 2, 5, 6};
  std::vector<int64_t> expected_indices = {0, 0, 0, 0};
  std::vector<int64_t> expected_dimensions = {2, 1, 2};
  int64_t axis = 1;
  RunTest(opset_version, 1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 0, sorted);
}

TEST(TopKOperator, Top1ExplicitAxisMultiDInputSmallestElements) {
  top_1_explicit_axis_MultiD_input_smallest<float>(11);
  top_1_explicit_axis_MultiD_input_smallest<float>(11, 0);  //unsorted
  top_1_explicit_axis_MultiD_input_smallest<double>(11);
  top_1_explicit_axis_MultiD_input_smallest<double>(11, 0);  //unsorted
  top_1_explicit_axis_MultiD_input_smallest<int32_t>(11);
  top_1_explicit_axis_MultiD_input_smallest<int32_t>(11, 0);  //unsorted
  top_1_explicit_axis_MultiD_input_smallest<int64_t>(11);
  top_1_explicit_axis_MultiD_input_smallest<int64_t>(11, 0);  //unsorted
}

// test path where SelectTopK is used (select using std::nth_element)
// we use a custom path for n=1, and priority queue based implementation if
//   bool use_priority_queue = k != 1 && (k < 4 || (std::log2(k) / std::log2(n)) < 0.725);
// so easiest way to test is for k to be 4 and n to be a little larger
TEST(TopKOperator, NthElement) {
  std::vector<float> input_vals = {10.0f, 8.0f, 7.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> input_dimensions = {6};
  std::vector<float> expected_vals = {10.0f, 8.0f, 7.0f, 6.0f};
  std::vector<int64_t> expected_indices = {0, 1, 2, 5};
  std::vector<int64_t> expected_dimensions = {4};
  RunTest(11, 4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, NthElement_double) {
  std::vector<double> input_vals = {10.0f, 8.0f, 7.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> input_dimensions = {6};
  std::vector<double> expected_vals = {10.0f, 8.0f, 7.0f, 6.0f};
  std::vector<int64_t> expected_indices = {0, 1, 2, 5};
  std::vector<int64_t> expected_dimensions = {4};
  RunTest(11, 4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, NthElementHalf) {
  if (!HasCudaEnvironment(600)) {
    return;
  }

  std::vector<float> input_vals_f = {10.0f, 8.0f, 7.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> expected_vals_f = {10.0f, 8.0f, 7.0f, 6.0f};
  std::vector<MLFloat16> input_vals(6);
  std::vector<MLFloat16> expected_vals(4);
  ConvertFloatToMLFloat16(input_vals_f.data(), input_vals.data(), 6);
  ConvertFloatToMLFloat16(expected_vals_f.data(), expected_vals.data(), 4);
  std::vector<int64_t> input_dimensions = {6};
  std::vector<int64_t> expected_indices = {0, 1, 2, 5};
  std::vector<int64_t> expected_dimensions = {4};
  RunTest(11, 4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

TEST(TopKOperator, NthElementHalf_NegtiveVals) {
  if (!HasCudaEnvironment(600)) {
    return;
  }

  std::vector<float> input_vals_f = {10.0f, -8.0f, -7.0f, -4.0f, -5.0f, -6.0f};
  std::vector<float> expected_vals_f = {10.0f, -4.0f, -5.0f, -6.0f};
  std::vector<MLFloat16> input_vals(6);
  std::vector<MLFloat16> expected_vals(4);
  ConvertFloatToMLFloat16(input_vals_f.data(), input_vals.data(), 6);
  ConvertFloatToMLFloat16(expected_vals_f.data(), expected_vals.data(), 4);
  std::vector<int64_t> input_dimensions = {6};
  std::vector<int64_t> expected_indices = {0, 3, 4, 5};
  std::vector<int64_t> expected_dimensions = {4};
  RunTest(11, 4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

// test dimension in range (GridDim::maxThreadsPerBlock, GridDim::maxThreadsPerBlock * 2], ie. [257, 512]
TEST(TopKOperator, SmallArrayTopKSorted) {
  std::vector<float> input_vals(400, 0.0f);
  std::iota(input_vals.begin(), input_vals.end(), 0.0f);
  std::vector<int64_t> input_dimensions = {400};
  std::vector<float> expected_vals(400, 0.0f);
  std::iota(expected_vals.begin(), expected_vals.end(), 0.0f);
  std::reverse(expected_vals.begin(), expected_vals.end());
  std::vector<int64_t> expected_indices(400, 0);
  std::iota(expected_indices.begin(), expected_indices.end(), 0);
  std::reverse(expected_indices.begin(), expected_indices.end());
  std::vector<int64_t> expected_dimensions = {400};
  RunTest(11, 400, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, -1, 1, 1);
}

TEST(TopKOperator, SmallArrayTopKSorted_double) {
  std::vector<double> input_vals(400, 0.0f);
  std::iota(input_vals.begin(), input_vals.end(), 0.0f);
  std::vector<int64_t> input_dimensions = {400};
  std::vector<double> expected_vals(400, 0.0f);
  std::iota(expected_vals.begin(), expected_vals.end(), 0.0f);
  std::reverse(expected_vals.begin(), expected_vals.end());
  std::vector<int64_t> expected_indices(400, 0);
  std::iota(expected_indices.begin(), expected_indices.end(), 0);
  std::reverse(expected_indices.begin(), expected_indices.end());
  std::vector<int64_t> expected_dimensions = {400};
  RunTest(11, 400, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, -1, 1, 1);
}

TEST(TopKOperator, MediumArrayTopKSorted) {
  std::vector<float> input_vals(1000, 0.0f);
  std::iota(input_vals.begin(), input_vals.end(), 0.0f);
  std::vector<int64_t> input_dimensions = {1000};
  std::vector<float> expected_vals(100, 0.0f);
  std::iota(expected_vals.begin(), expected_vals.end(), 900.0f);
  std::reverse(expected_vals.begin(), expected_vals.end());
  std::vector<int64_t> expected_indices(100, 0);
  std::iota(expected_indices.begin(), expected_indices.end(), 900);
  std::reverse(expected_indices.begin(), expected_indices.end());
  std::vector<int64_t> expected_dimensions = {100};
  RunTest(11, 100, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, 0, 1, 1);
}

TEST(TopKOperator, MediumArrayTopKSorted_double) {
  std::vector<double> input_vals(1000, 0.0f);
  std::iota(input_vals.begin(), input_vals.end(), 0.0f);
  std::vector<int64_t> input_dimensions = {1000};
  std::vector<double> expected_vals(100, 0.0f);
  std::iota(expected_vals.begin(), expected_vals.end(), 900.0f);
  std::reverse(expected_vals.begin(), expected_vals.end());
  std::vector<int64_t> expected_indices(100, 0);
  std::iota(expected_indices.begin(), expected_indices.end(), 900);
  std::reverse(expected_indices.begin(), expected_indices.end());
  std::vector<int64_t> expected_dimensions = {100};
  RunTest(11, 100, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, 0, 1, 1);
}

TEST(TopKOperator, BigArrayTopKSorted) {
  std::vector<float> input_vals(10000, 0.0f);
  std::iota(input_vals.begin(), input_vals.end(), 0.0f);
  std::vector<int64_t> input_dimensions = {10000};
  std::vector<float> expected_vals(1000, 0.0f);
  std::iota(expected_vals.begin(), expected_vals.end(), 9000.0f);
  std::reverse(expected_vals.begin(), expected_vals.end());
  std::vector<int64_t> expected_indices(1000, 0);
  std::iota(expected_indices.begin(), expected_indices.end(), 9000);
  std::reverse(expected_indices.begin(), expected_indices.end());
  std::vector<int64_t> expected_dimensions = {1000};
  RunTest(11, 1000, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, 0, 1, 1);
}

TEST(TopKOperator, BigArrayTopKSorted_double) {
  std::vector<double> input_vals(10000, 0.0f);
  std::iota(input_vals.begin(), input_vals.end(), 0.0f);
  std::vector<int64_t> input_dimensions = {10000};
  std::vector<double> expected_vals(1000, 0.0f);
  std::iota(expected_vals.begin(), expected_vals.end(), 9000.0f);
  std::reverse(expected_vals.begin(), expected_vals.end());
  std::vector<int64_t> expected_indices(1000, 0);
  std::iota(expected_indices.begin(), expected_indices.end(), 9000);
  std::reverse(expected_indices.begin(), expected_indices.end());
  std::vector<int64_t> expected_dimensions = {1000};
  RunTest(11, 1000, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, 0, 1, 1);
}

TEST(TopKOperator, BigArrayBigTopKSorted) {
  std::vector<float> input_vals(10000, 0.0f);
  std::iota(input_vals.begin(), input_vals.end(), 0.0f);
  std::vector<int64_t> input_dimensions = {10000};
  std::vector<float> expected_vals(9000, 0.0f);
  std::iota(expected_vals.begin(), expected_vals.end(), 1000.0f);
  std::reverse(expected_vals.begin(), expected_vals.end());
  std::vector<int64_t> expected_indices(9000, 0);
  std::iota(expected_indices.begin(), expected_indices.end(), 1000);
  std::reverse(expected_indices.begin(), expected_indices.end());
  std::vector<int64_t> expected_dimensions = {9000};
  RunTest(11, 9000, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, 0, 1, 1);
}

TEST(TopKOperator, BigArrayBigTopKSorted_double) {
  std::vector<double> input_vals(10000, 0.0f);
  std::iota(input_vals.begin(), input_vals.end(), 0.0f);
  std::vector<int64_t> input_dimensions = {10000};
  std::vector<double> expected_vals(9000, 0.0f);
  std::iota(expected_vals.begin(), expected_vals.end(), 1000.0f);
  std::reverse(expected_vals.begin(), expected_vals.end());
  std::vector<int64_t> expected_indices(9000, 0);
  std::iota(expected_indices.begin(), expected_indices.end(), 1000);
  std::reverse(expected_indices.begin(), expected_indices.end());
  std::vector<int64_t> expected_dimensions = {9000};
  RunTest(11, 9000, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, 0, 1, 1);
}

template <typename T>
static void top_3_all_same(int opset_version, int64_t largest = 1) {
  // whether it's largest or smallest we should pick the first instance/s of a number if there are multiple
  std::vector<T> input_vals = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<T> expected_vals = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
  std::vector<int64_t> expected_indices = {0, 1, 2, 0, 1, 2};
  std::vector<int64_t> expected_dimensions = {2, 3};
  RunTest(opset_version, 3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, -1, largest);
}

TEST(TopKOperator, Top3AllSame) {
  top_3_all_same<float>(10);
  top_3_all_same<float>(11);
  top_3_all_same<float>(10, 0);  // smallest
  top_3_explicit_axis<float>(11, 0);
  top_3_all_same<double>(10);
  top_3_all_same<double>(11);
  top_3_all_same<double>(10, 0);  // smallest
  top_3_explicit_axis<double>(11, 0);
}

template <typename T>
static void TestThreaded(int64_t k, int64_t n, int64_t batch_size) {
  std::vector<T> input_vals(n * batch_size, 0.0f);
  std::iota(input_vals.begin(), input_vals.end(), 0.0f);

  std::vector<int64_t> input_dimensions = {n, batch_size};

  std::vector<T> expected_vals(n * k, 0.0f);
  std::vector<int64_t> expected_indices(n * k, 0);
  std::vector<int64_t> expected_dimensions = {n, k};

  for (int64_t i = 0; i < n; ++i) {
    auto begin_batch_output = expected_vals.begin() + i * k;
    std::iota(begin_batch_output, begin_batch_output + k, static_cast<T>(((i + 1) * batch_size) - k));
    std::reverse(begin_batch_output, begin_batch_output + k);

    // indices are within the axis so don't need adjusting by the batch number
    auto begin_indices_output = expected_indices.begin() + i * k;
    std::iota(begin_indices_output, begin_indices_output + k, batch_size - k);
    std::reverse(begin_indices_output, begin_indices_output + k);
  }

  RunTest(11, k, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false);
}

// create input of 2x1000 and select 200 so 2 threads are needed based on there being 2 rows
// and sufficient items to process given this calculation:
//   int64_t threads_needed = static_cast<int64_t>(std::floor(input_shape.Size() * k / (128 * 1024)));
TEST(TopKOperator, PriorityQueueThreaded) {
  const int64_t k = 200;
  const int64_t n = 2;
  const int64_t batch_size = 1000;
  TestThreaded<float>(k, n, batch_size);
  TestThreaded<double>(k, n, batch_size);
}

// create input of 2x500 and select 400 so 2 threads are needed based on there being 2 rows
// and sufficient items to process given this calculation:
//   int64_t threads_needed = static_cast<int64_t>(std::floor(input_shape.Size() * k / (128 * 1024)));
TEST(TopKOperator, SelectTopKThreaded) {
  const int64_t k = 400;
  const int64_t n = 2;
  const int64_t batch_size = 500;
  TestThreaded<float>(k, n, batch_size);
  TestThreaded<double>(k, n, batch_size);
}

}  // namespace test
}  // namespace onnxruntime
