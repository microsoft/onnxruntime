// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <numeric>

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {
void RunExpandDimsTest(const std::vector<int64_t>& input_shape, const int32_t expand_axis,
                       const std::vector<int64_t>& expected_output_shape) {
  const auto input_size = std::accumulate(input_shape.begin(), input_shape.end(),
                                          int64_t{1}, std::multiplies<>{});
  const auto input_data = std::vector<float>(input_size, 2.0f);

  OpTester test("ExpandDims", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("X", input_shape, input_data);
  test.AddInput<int32_t>("axis", {}, {expand_axis});
  test.AddOutput<float>("Y", expected_output_shape, input_data);
  test.Run();
}

void RunExpandDimsExpectedFailureTest(const std::vector<int64_t>& input_shape, const int32_t expand_axis,
                                      const std::string& expected_failure_message) {
  const auto input_size = std::accumulate(input_shape.begin(), input_shape.end(),
                                          int64_t{1}, std::multiplies<>{});
  const auto input_data = std::vector<float>(input_size, 2.0f);

  OpTester test("ExpandDims", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("X", input_shape, input_data);
  test.AddInput<int32_t>("axis", {}, {expand_axis});
  test.AddOutput<float>("Y", input_shape, input_data);
  test.Run(BaseTester::ExpectResult::kExpectFailure, expected_failure_message);
}
}  // namespace

TEST(ExpandDimsTest, Basic) {
  RunExpandDimsTest({2, 3}, 1, {2, 1, 3});
}

TEST(ExpandDimsTest, NegativeAxis) {
  RunExpandDimsTest({2, 3}, -1, {2, 3, 1});
}

TEST(ExpandDimsTest, MaxAxis) {
  RunExpandDimsTest({2, 3}, 2, {2, 3, 1});
}

TEST(ExpandDimsTest, MinAxis) {
  RunExpandDimsTest({2, 3}, -3, {1, 2, 3});
}

TEST(ExpandDimsTest, PositiveAxisOutOfRange) {
  RunExpandDimsExpectedFailureTest({2, 3}, 3, "Axis must be within range [-3, 2]");
}

TEST(ExpandDimsTest, NegativeAxisOutOfRange) {
  RunExpandDimsExpectedFailureTest({2, 3}, -4, "Axis must be within range [-3, 2]");
}

}  // namespace test
}  // namespace onnxruntime
