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

// Runs ExpandDims with the axis supplied as a constant initializer. This causes the
// operator's shape-inference function (which only runs when the axis value is known at
// graph-resolution time via getInputData) to be exercised, in addition to the kernel.
void RunExpandDimsConstAxisTest(const std::vector<int64_t>& input_shape, const int32_t expand_axis,
                                const std::vector<int64_t>& expected_output_shape) {
  SCOPED_TRACE(MakeString("input_shape: ", TensorShape(input_shape), ", expand_axis: ", expand_axis));

  const auto input_size = ShapeSize(input_shape);
  const auto input_data = GenerateInput(input_size);

  OpTester test("ExpandDims", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("X", input_shape, input_data);
  test.AddInput<int32_t>("axis", {}, {expand_axis}, /*is_initializer=*/true);
  test.AddOutput<float>("Y", expected_output_shape, input_data);
  test.Run();
}

void RunExpandDimsMalformedConstAxisTest(const std::vector<int64_t>& input_shape,
                                         const std::vector<int32_t>& expand_axes,
                                         const std::string& expected_failure_message) {
  SCOPED_TRACE(MakeString("input_shape: ", TensorShape(input_shape)));

  const auto input_size = ShapeSize(input_shape);
  const auto input_data = GenerateInput(input_size);

  OpTester test("ExpandDims", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("X", input_shape, input_data);
  test.AddInput<int32_t>("axis", {static_cast<int64_t>(expand_axes.size())}, expand_axes,
                         /*is_initializer=*/true);
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

// Regression test for a heap out-of-bounds read in the ExpandDims shape-inference function.
// When the axis is a constant initializer, shape inference runs during graph resolution and
// normalizes a negative axis against the output rank. The previous formula produced a negative
// dimension index for the most-negative axes (e.g. pos == -2), indexing before the start of the
// protobuf RepeatedPtrField backing store. These cases must resolve to valid output shapes.
TEST(ExpandDimsTest, NegativeAxisConstInitializerShapeInference) {
  RunExpandDimsConstAxisTest({2, 3}, -1, {2, 3, 1});
  RunExpandDimsConstAxisTest({2, 3}, -2, {2, 1, 3});
  RunExpandDimsConstAxisTest({2, 3}, -3, {1, 2, 3});  // minimum valid axis; previously read out of bounds
  RunExpandDimsConstAxisTest({}, -1, {1});            // scalar, minimum valid axis; previously read out of bounds
}

#ifndef ORT_NO_EXCEPTIONS
TEST(ExpandDimsTest, MalformedConstInitializerAxisFailsShapeInference) {
  RunExpandDimsMalformedConstAxisTest({2, 3}, {0, 1},
                                      "Input axis must be a single int32 scalar initializer");
}
#endif  // !ORT_NO_EXCEPTIONS

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
