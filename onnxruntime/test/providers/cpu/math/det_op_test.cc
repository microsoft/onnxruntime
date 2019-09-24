// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(DetOpTest, 2d) {
  OpTester test("Det", 11);
  test.AddInput<float>("X", {2, 2}, {0., 1., 2., 3.});
  test.AddOutput<float>("Y", {}, {-2.});
  test.Run();
}

TEST(DetOpTest, 2dWithBatchSize3) {
  OpTester test("Det", 11);
  test.AddInput<float>("X", {3, 2, 2}, {1., 2., 3., 4., 1., 2., 2., 1., 1., 3., 3., 1.});
  test.AddOutput<float>("Y", {3}, {-2., -3., -8.});
  test.Run();
}

TEST(DetOpTest, 2dWithMultipleBatchDims) {
  OpTester test("Det", 11);
  test.AddInput<float>("X",
                       {1, 2, 3, 2, 2},
                       {1., 2., 3., 4., 1., 2., 2., 1., 1., 3., 3., 1., 1., 2., 3., 4., 1., 2., 2., 1., 1., 3., 3., 1.});
  test.AddOutput<float>("Y", {1, 2, 3}, {-2., -3., -8., -2., -3., -8.});
  test.Run();
}

TEST(DetOpTest, InputDimsLessThan2) {
  OpTester test("Det", 11);
  test.AddInput<float>("X", {1}, {3.0f});
  test.AddOutput<float>("Y", {}, {-14.});
  test.Run(OpTester::ExpectResult::kExpectFailure, "[ShapeInferenceError] Input rank must be >= 2.");
}

TEST(DetOpTest, InputDoesNotHaveSquareMatrix) {
  OpTester test("Det", 11);
  test.AddInput<float>("X", {2, 3}, {3.0f, 8.0f, 4.0f, 6.0f, 4.0f, 6.0f});
  test.AddOutput<float>("Y", {}, {-14.});
  test.Run(OpTester::ExpectResult::kExpectFailure, "[ShapeInferenceError] The inner-most 2 dimensions must have the same size");
}

}  // namespace test
}  // namespace onnxruntime
