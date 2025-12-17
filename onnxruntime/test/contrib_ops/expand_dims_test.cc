// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <numeric>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/tensor_shape.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {
size_t ShapeSize(const std::vector<int64_t>& shape) {
  return narrow<size_t>(TensorShape(shape).Size());
}

std::vector<float> GenerateInput(size_t size) {
  std::vector<float> result(size);
  std::iota(result.begin(), result.end(), 1.0f);
  return result;
}

void RunExpandDimsTest(const std::vector<int64_t>& input_shape, const int32_t expand_axis,
                       const std::vector<int64_t>& expected_output_shape) {
  SCOPED_TRACE(MakeString("input_shape: ", TensorShape(input_shape), ", expand_axis: ", expand_axis));

  const auto input_size = ShapeSize(input_shape);
  const auto input_data = GenerateInput(input_size);

  OpTester test("ExpandDims", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("X", input_shape, input_data);
  test.AddInput<int32_t>("axis", {}, {expand_axis});
  test.AddOutput<float>("Y", expected_output_shape, input_data);
  test.Run();
}

void RunExpandDimsExpectedFailureTest(const std::vector<int64_t>& input_shape, const int32_t expand_axis,
                                      const std::string& expected_failure_message) {
  SCOPED_TRACE(MakeString("input_shape: ", TensorShape(input_shape), ", expand_axis: ", expand_axis));

  const auto input_size = ShapeSize(input_shape);
  const auto input_data = GenerateInput(input_size);

  OpTester test("ExpandDims", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("X", input_shape, input_data);
  test.AddInput<int32_t>("axis", {}, {expand_axis});
  test.AddOutput<float>("Y", input_shape, input_data);
  test.Run(BaseTester::ExpectResult::kExpectFailure, expected_failure_message);
}
}  // namespace

TEST(ExpandDimsTest, Basic) {
  RunExpandDimsTest({2, 3}, 1, {2, 1, 3});
  RunExpandDimsTest({2, 3}, 0, {1, 2, 3});
  RunExpandDimsTest({2, 3}, -1, {2, 3, 1});
}

TEST(ExpandDimsTest, MaxAxis) {
  RunExpandDimsTest({2, 3}, 2, {2, 3, 1});
  RunExpandDimsTest({}, 0, {1});
}

TEST(ExpandDimsTest, MinAxis) {
  RunExpandDimsTest({2, 3}, -3, {1, 2, 3});
  RunExpandDimsTest({}, -1, {1});
}

TEST(ExpandDimsTest, PositiveAxisOutOfRange) {
  RunExpandDimsExpectedFailureTest({2, 3}, 3, "Axis must be within range [-3, 2]. Axis is 3");
  RunExpandDimsExpectedFailureTest({}, 1, "Axis must be within range [-1, 0]. Axis is 1");
}

TEST(ExpandDimsTest, NegativeAxisOutOfRange) {
  RunExpandDimsExpectedFailureTest({2, 3}, -4, "Axis must be within range [-3, 2]. Axis is -4");
  RunExpandDimsExpectedFailureTest({}, -2, "Axis must be within range [-1, 0]. Axis is -2");
}

}  // namespace test
}  // namespace onnxruntime
