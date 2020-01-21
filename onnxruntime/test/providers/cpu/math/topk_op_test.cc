// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

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
          1,
          1,
          OpTester::ExpectResult::kExpectFailure,
          "Invalid value for attribute k");
}

static void top_0_default_axis(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {};
  std::vector<int64_t> expected_indices = {};
  std::vector<int64_t> expected_dimensions = {2, 0};
  RunTest(opset_version, 0, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, -1, 1, sorted);
}

TEST(TopKOperator, Top0DefaultAxisLargestElements) {
  top_0_default_axis(10);
  top_0_default_axis(11);
  top_0_default_axis(11, 0);  // unsorted
}

static void top_1_default_axis(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1};
  std::vector<int64_t> expected_dimensions = {2, 1};
  int64_t axis = -1;
  RunTest(opset_version, 1, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top1DefaultAxisLargestElements) {
  top_1_default_axis(10);
  top_1_default_axis(11);
  top_1_default_axis(11, 0);  // unsorted
}

static void top_2_default_axis(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.4f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 1};
  std::vector<int64_t> expected_dimensions = {2, 2};
  int64_t axis = -1;
  RunTest(opset_version, 2, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top2DefaultAxisLargestElements) {
  top_2_default_axis(10);
  top_2_default_axis(11);
  top_2_default_axis(11, 0);  // unsorted
}

static void top_3_default_axis(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.2f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 2, 1, 3};
  std::vector<int64_t> expected_dimensions = {2, 3};
  int64_t axis = -1;
  RunTest(opset_version, 3, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top3DefaultAxisLargestElements) {
  top_3_default_axis(10);
  top_3_default_axis(11);
  top_3_default_axis(11, 0);  //unsorted
}

static void top_all_default_axis(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.4f, 0.3f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f};
  std::vector<int64_t> expected_indices = {3, 1, 2, 0, 1, 2, 3, 0};
  std::vector<int64_t> expected_dimensions = {2, 4};
  int64_t axis = -1;
  RunTest(opset_version, 4, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, TopAllDefaultAxisLargestElements) {
  top_all_default_axis(10);
  top_all_default_axis(11);
  top_all_default_axis(11, 0);  // unsorted
}

static void top_1_explicit_axis(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.3f, 0.4f};
  std::vector<int64_t> expected_indices = {3, 1};
  std::vector<int64_t> expected_dimensions = {1, 2};
  int64_t axis = 0;
  RunTest(opset_version, 1, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top1ExplicitAxisLargestElements) {
  top_1_explicit_axis(10);
  top_1_explicit_axis(11);
  top_1_explicit_axis(11, 0);  // unsorted
}

static void top_2_explicit_axis(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.0f, 1.0f, 2.0f, 11.0f, 08.0f, 5.0f, 6.0f, 7.0f, 4.0f, 9.0f, 10.0f, 3.0f};
  std::vector<int64_t> input_dimensions = {3, 4};
  std::vector<float> expected_vals = {8.0f, 9.0f, 10.0f, 11.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<int64_t> expected_indices = {1, 2, 2, 0, 2, 1, 1, 1};
  std::vector<int64_t> expected_dimensions = {2, 4};
  int64_t axis = 0;
  RunTest(opset_version, 2, input_vals, input_dimensions, expected_vals, expected_indices,
          expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top2ExplicitAxisLargestElements) {
  top_2_explicit_axis(10);
  top_2_explicit_axis(11);
  top_2_explicit_axis(11, 0);  //unsorted
}

static void top_3_explicit_axis(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.3f, 0.4f, 0.2f, 0.3f, 0.1f, 0.3f};
  std::vector<int64_t> expected_indices = {3, 1, 1, 0, 0, 2};
  std::vector<int64_t> expected_dimensions = {3, 2};
  int64_t axis = 0;
  RunTest(opset_version, 3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top3ExplicitAxisLargestElements) {
  top_3_explicit_axis(10);
  top_3_explicit_axis(11);
  top_3_explicit_axis(11, 0);  //unsorted
}

static void top_all_explicit_axis(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.3f, 0.4f, 0.2f, 0.3f, 0.1f, 0.3f, 0.1f, 0.2f};
  std::vector<int64_t> expected_indices = {3, 1, 1, 0, 0, 2, 2, 3};
  std::vector<int64_t> expected_dimensions = {4, 2};
  int64_t axis = 0;
  RunTest(opset_version, 4, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, TopAllExplicitAxisLargestElements) {
  top_all_explicit_axis(10);
  top_all_explicit_axis(11);
  top_all_explicit_axis(11, 0);  // unsorted
}

static void top_all_explicit_axis_1D_input(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f, 285.0f, 527.0f, 862.0f};
  std::vector<int64_t> input_dimensions = {13};
  std::vector<float> expected_vals = {983.0f, 978.0f, 971.0f, 862.0f, 723.0f, 695.0f, 531.0f, 527.0f, 483.0f, 285.0f, 247.0f, 242.0f, 93.0f};
  std::vector<int64_t> expected_indices = {7, 3, 2, 12, 9, 1, 8, 11, 4, 10, 5, 6, 0};
  std::vector<int64_t> expected_dimensions = {13};
  int64_t axis = 0;
  RunTest(opset_version, 13, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, TopAllExplicitAxis1DInputLargestElements) {
  top_all_explicit_axis_1D_input(10);
  top_all_explicit_axis_1D_input(11);
  top_all_explicit_axis_1D_input(11, 0);  // unsorted
}

static void top_2_explicit_axis_1D_large_input(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {93.0f, 695.0f, 971.0f, 978.0f, 483.0f, 247.0f, 242.0f, 983.0f, 531.0f, 723.0f,
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
  std::vector<float> expected_vals = {983.0f, 983.0f};
  std::vector<int64_t> expected_indices = {7, 17};
  std::vector<int64_t> expected_dimensions = {2};
  int64_t axis = 0;
  RunTest(opset_version, 2, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, TopAllExplicitAxis1DLargeInputLargestElements) {
  top_2_explicit_axis_1D_large_input(10);
  top_2_explicit_axis_1D_large_input(11);
  top_2_explicit_axis_1D_large_input(11, 0);  // unsorted
}

static void top_1_explicit_axis_MultiD_input(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<int64_t> input_dimensions = {2, 2, 2};
  std::vector<float> expected_vals = {3, 4, 7, 8};
  std::vector<int64_t> expected_indices = {1, 1, 1, 1};
  std::vector<int64_t> expected_dimensions = {2, 1, 2};
  int64_t axis = 1;
  RunTest(opset_version, 1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 1, sorted);
}

TEST(TopKOperator, Top1ExplicitAxisMultiDInputLargestElements) {
  top_1_explicit_axis_MultiD_input(10);
  top_1_explicit_axis_MultiD_input(11);
  top_1_explicit_axis_MultiD_input(11, 0);  // unsorted
}

static void top_2_default_axis_smallest(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.4f, 0.2f};
  std::vector<int64_t> input_dimensions = {2, 4};
  std::vector<float> expected_vals = {0.1f, 0.2f, 0.1f, 0.2f};
  std::vector<int64_t> expected_indices = {0, 2, 0, 3};
  std::vector<int64_t> expected_dimensions = {2, 2};
  int64_t axis = -1;
  RunTest(opset_version, 2, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 0, sorted);
}

TEST(TopKOperator, Top2DefaultAxisSmallestElements) {
  top_2_default_axis_smallest(11);
  top_2_default_axis_smallest(11, 0);  // unsorted
}

static void top_3_explicit_axis_smallest(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {0.1f, 0.3f, 0.2f, 0.4f, 0.1f, 0.3f, 0.3f, 0.2f};
  std::vector<int64_t> input_dimensions = {4, 2};
  std::vector<float> expected_vals = {0.1f, 0.2f, 0.1f, 0.3f, 0.2f, 0.3f};
  std::vector<int64_t> expected_indices = {0, 3, 2, 0, 1, 2};
  std::vector<int64_t> expected_dimensions = {3, 2};
  int64_t axis = 0;
  RunTest(opset_version, 3, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 0, sorted);
}

TEST(TopKOperator, Top3ExplicitAxisSmallestElements) {
  top_3_explicit_axis_smallest(11);
  top_3_explicit_axis_smallest(11, 0);  //unsorted
}

static void top_1_explicit_axis_MultiD_input_smallest(int opset_version, int64_t sorted = 1) {
  std::vector<float> input_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<int64_t> input_dimensions = {2, 2, 2};
  std::vector<float> expected_vals = {1, 2, 5, 6};
  std::vector<int64_t> expected_indices = {0, 0, 0, 0};
  std::vector<int64_t> expected_dimensions = {2, 1, 2};
  int64_t axis = 1;
  RunTest(opset_version, 1, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 0, sorted);
}

TEST(TopKOperator, Top1ExplicitAxisMultiDInputSmallestElements) {
  top_1_explicit_axis_MultiD_input_smallest(11);
  top_1_explicit_axis_MultiD_input_smallest(11, 0);  //unsorted
}

TEST(TopKOperator, SelectFirstSortNext) {
  // in this test, we will select the top 5 elements first then sort the chosen 5 elements
  // Select + Sort  = O(n + k * ln(k)) = 50 + 5 * ln(5) = 58.047
  // Sorted selection: O(n * ln(k)) = 50 * ln(5) = 80.47
  // The algorithm used will be Select + Sort
  std::vector<float> input_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0,
                                   11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0,
                                   21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0,
                                   31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0,
                                   41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0};
  std::vector<int64_t> input_dimensions = {50};
  std::vector<float> expected_vals = {50.0f, 49.0f, 48.0f, 47.0f, 46.0f};
  std::vector<int64_t> expected_indices = {49, 48, 47, 46, 45};
  std::vector<int64_t> expected_dimensions = {5};
  int64_t axis = 0;
  RunTest(11, 5, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);  // largest values
}

TEST(TopKOperator, SelectFirstSortNextInt64) {
  // in this test, we will select the top 5 elements first then sort the chosen 5 elements
  // Select + Sort  = O(n + k * ln(k)) = 50 + 5 * ln(5) = 58.047
  // Sorted selection: O(n * ln(k)) = 50 * ln(5) = 80.47
  // The algorithm used will be Select + Sort
  std::vector<int64_t> input_vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
  std::vector<int64_t> input_dimensions = {50};
  std::vector<int64_t> expected_vals = {50, 49, 48, 47, 46};
  std::vector<int64_t> expected_indices = {49, 48, 47, 46, 45};
  std::vector<int64_t> expected_dimensions = {5};
  int64_t axis = 0;
  RunTest(11, 5, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis);  // largest values
}

TEST(TopKOperator, SortedSelection) {
  // in this test, we will use sorted selection (using heap)
  // Select + Sort  = O(n + k * ln(k)) = 10 + 5 * ln(5) = 18.04
  // Sorted selection: O(n * ln(k)) = 10 * ln(5) = 16.09
  // The algorithm used will be Sorted selection
  std::vector<float> input_vals = {10.0f, 8.0f, 7.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 9.0f, 3.0};
  std::vector<int64_t> input_dimensions = {10};
  std::vector<float> expected_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<int64_t> expected_indices = {6, 7, 9, 3, 4};
  std::vector<int64_t> expected_dimensions = {5};
  int64_t axis = 0;
  RunTest(11, 5, input_vals, input_dimensions, expected_vals, expected_indices, expected_dimensions, false, axis, 0);  // smallest values
}

TEST(TopKOperator, MediumArrayTopKSorted) 
{
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

TEST(TopKOperator, BigArrayTopKSorted) 
{
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

TEST(TopKOperator, BigArrayBigTopKSorted) 
{
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

}  // namespace test
}  // namespace onnxruntime
