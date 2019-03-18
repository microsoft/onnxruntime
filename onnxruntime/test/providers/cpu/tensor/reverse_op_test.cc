// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

template<typename T>
static void RunTest(const std::vector<T>& input_vals,
					const std::vector<int64_t> input_dimensions,
                    const std::vector<int64_t>& reverse_axes,
                    const std::vector<T>& expected_vals,
                    const std::vector<int64_t> expected_dimensions,
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                    const std::string& expected_err_str = "") {
  OpTester test("Reverse", 10);

  // Attribute
  if (reverse_axes.size() != 0)
    test.AddAttribute("input", reverse_axes);

  // Input
  test.AddInput<T>("input", input_vals, input_dimensions);

  // Output
  test.AddOutput<T>("output", expected_vals, expected_dimensions);
  
  // Run test and check results
  test.Run(expect_result, expected_err_str);
}

template <typename T>
static void RunTestWrapper()
{
	// Test 1 (scalar)
	std::vector<T> input_vals_1 = {1};
	std::vector<int64_t> input_dimensions_1;
	std::vector<int64_t> reverse_axes_1;
	std::vector<T> expected_vals_1 = {1};
	std::vector<int64_t> expected_dimensions_1;
	RunTest<T>(input_vals_1, input_dimensions_1, reverse_axes_1, expected_vals_1, expected_dimensions_1);

	// Test 2 (default axes)
    std::vector<T> input_vals_2 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> input_dimensions_2 = {2, 4};
    std::vector<int64_t> reverse_axes_2;
    std::vector<T> expected_vals_2 = {8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<int64_t> expected_dimensions_2 = {2, 4};
    RunTest<T>(input_vals_2, input_dimensions_2, reverse_axes_2, expected_vals_2, expected_dimensions_2);

	// Test 3 (explicit axes)
    std::vector<T> input_vals_3 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> input_dimensions_3 = {2, 4};
    std::vector<int64_t> reverse_axes_3 = {0, 1};
    std::vector<T> expected_vals_3 = {8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<int64_t> expected_dimensions_3 = {2, 4};
    RunTest<T>(input_vals_3, input_dimensions_3, reverse_axes_3, expected_vals_3, expected_dimensions_3);

	// Test 4 (explicit axes with negative axis)
    std::vector<T> input_vals_4 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> input_dimensions_4 = {2, 4};
    std::vector<int64_t> reverse_axes_4 = {-1, 0};
    std::vector<T> expected_vals_4 = {8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<int64_t> expected_dimensions_4 = {2, 4};
    RunTest<T>(input_vals_4, input_dimensions_4, reverse_axes_4, expected_vals_4, expected_dimensions_4);
}

template <>
static void RunTestWrapper<bool>() {
  // Test 1 (scalar)
  std::vector<bool> input_vals_1 = {true};
  std::vector<int64_t> input_dimensions_1;
  std::vector<int64_t> reverse_axes_1;
  std::vector<bool> expected_vals_1 = {true};
  std::vector<int64_t> expected_dimensions_1;
  RunTest<bool>(input_vals_1, input_dimensions_1, reverse_axes_1, expected_vals_1, expected_dimensions_1);

  // Test 2 (default axes)
  std::vector<bool> input_vals_2 = {true, true, false, false, true, true, false, false};
  std::vector<int64_t> input_dimensions_2 = {2, 4};
  std::vector<int64_t> reverse_axes_2;
  std::vector<bool> expected_vals_2 = {false, false, true, true, false, false, true, true};
  std::vector<int64_t> expected_dimensions_2 = {2, 4};
  RunTest<bool>(input_vals_2, input_dimensions_2, reverse_axes_2, expected_vals_2, expected_dimensions_2);

  // Test 3 (explicit axes)
  std::vector<bool> input_vals_3 = {true, true, false, false, true, true, false, false};
  std::vector<int64_t> input_dimensions_3 = {2, 4};
  std::vector<int64_t> reverse_axes_3 = {0, 1};
  std::vector<bool> expected_vals_3 = {false, false, true, true, false, false, true, true};
  std::vector<int64_t> expected_dimensions_3 = {2, 4};
  RunTest<bool>(input_vals_3, input_dimensions_3, reverse_axes_3, expected_vals_3, expected_dimensions_3);

  // Test 4 (explicit axes with negative axis)
  std::vector<bool> input_vals_4 = {true, true, false, false, true, true, false, false};
  std::vector<int64_t> input_dimensions_4 = {2, 4};
  std::vector<int64_t> reverse_axes_4 = {-1, 0};
  std::vector<bool> expected_vals_4 = {false, false, true, true, false, false, true, true};
  std::vector<int64_t> expected_dimensions_4 = {2, 4};
  RunTest<bool>(input_vals_4, input_dimensions_4, reverse_axes_4, expected_vals_4, expected_dimensions_4);
}

template <>
static void RunTestWrapper<std::string>() {
  // Test 1 (scalar)
  std::vector<std::string> input_vals_1 = {"a"};
  std::vector<int64_t> input_dimensions_1;
  std::vector<int64_t> reverse_axes_1;
  std::vector<std::string> expected_vals_1 = {"a"};
  std::vector<int64_t> expected_dimensions_1;
  RunTest<std::string>(input_vals_1, input_dimensions_1, reverse_axes_1, expected_vals_1, expected_dimensions_1);

  // Test 2 (default axes)
  std::vector<std::string> input_vals_2 = {"a", "b", "c", "d", "e", "f", "g", "h"};
  std::vector<int64_t> input_dimensions_2 = {2, 4};
  std::vector<int64_t> reverse_axes_2;
  std::vector<std::string> expected_vals_2 = {"h", "g", "f", "e", "d", "c", "b", "a"};
  std::vector<int64_t> expected_dimensions_2 = {2, 4};
  RunTest<std::string>(input_vals_2, input_dimensions_2, reverse_axes_2, expected_vals_2, expected_dimensions_2);

  // Test 3 (explicit axes)
  std::vector<std::string> input_vals_3 = {"a", "b", "c", "d", "e", "f", "g", "h"};
  std::vector<int64_t> input_dimensions_3 = {2, 4};
  std::vector<int64_t> reverse_axes_3 = {0, 1};
  std::vector<std::string> expected_vals_3 = {"h", "g", "f", "e", "d", "c", "b", "a"};
  std::vector<int64_t> expected_dimensions_3 = {2, 4};
  RunTest<std::string>(input_vals_3, input_dimensions_3, reverse_axes_3, expected_vals_3, expected_dimensions_3);

  // Test 4 (explicit axes with negative axis)
  std::vector<std::string> input_vals_4 = {"a", "b", "c", "d", "e", "f", "g", "h"};
  std::vector<int64_t> input_dimensions_4 = {2, 4};
  std::vector<int64_t> reverse_axes_4 = {-1, 0};
  std::vector<std::string> expected_vals_4 = {"h", "g", "f", "e", "d", "c", "b", "a"};
  std::vector<int64_t> expected_dimensions_4 = {2, 4};
  RunTest<std::string>(input_vals_4, input_dimensions_4, reverse_axes_4, expected_vals_4, expected_dimensions_4);
}

static const std::vector<MLFloat16> ConvertFloatToMLFloat16(const std::vector<float>& float_data) {
  std::vector<MLFloat16> new_data;
  for (const auto& f : float_data) {
    new_data.push_back(MLFloat16(math::floatToHalf(f)));
  }
  return new_data;
}

template <>
static void RunTestWrapper<MLFloat16>() {
  // Test 1 (scalar)
  std::vector<float> input_vals_1 = {1};
  std::vector<int64_t> input_dimensions_1;
  std::vector<int64_t> reverse_axes_1;
  std::vector<float> expected_vals_1 = {1};
  std::vector<int64_t> expected_dimensions_1;
  RunTest<MLFloat16>(ConvertFloatToMLFloat16(input_vals_1), input_dimensions_1, reverse_axes_1, ConvertFloatToMLFloat16(expected_vals_1), expected_dimensions_1);

  // Test 2 (default axes)
  std::vector<float> input_vals_2 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> input_dimensions_2 = {2, 4};
  std::vector<int64_t> reverse_axes_2;
  std::vector<float> expected_vals_2 = {8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int64_t> expected_dimensions_2 = {2, 4};
  RunTest<MLFloat16>(ConvertFloatToMLFloat16(input_vals_2), input_dimensions_2, reverse_axes_2, ConvertFloatToMLFloat16(expected_vals_2), expected_dimensions_2);

  // Test 3 (explicit axes)
  std::vector<float> input_vals_3 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> input_dimensions_3 = {2, 4};
  std::vector<int64_t> reverse_axes_3 = {0, 1};
  std::vector<float> expected_vals_3 = {8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int64_t> expected_dimensions_3 = {2, 4};
  RunTest<MLFloat16>(ConvertFloatToMLFloat16(input_vals_3), input_dimensions_3, reverse_axes_3, ConvertFloatToMLFloat16(expected_vals_3), expected_dimensions_3);

  // Test 4 (explicit axes with negative axis)
  std::vector<float> input_vals_4 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> input_dimensions_4 = {2, 4};
  std::vector<int64_t> reverse_axes_4 = {-1, 0};
  std::vector<float> expected_vals_4 = {8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int64_t> expected_dimensions_4 = {2, 4};
  RunTest<MLFloat16>(ConvertFloatToMLFloat16(input_vals_4), input_dimensions_4, reverse_axes_4, ConvertFloatToMLFloat16(expected_vals_4), expected_dimensions_4);
}
}  // namespace test
}  // namespace onnxruntime