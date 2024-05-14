// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/trt_op_test_utils.h"

using namespace std;

namespace onnxruntime {
namespace test {

TEST(OneHotOpTest, DefaultAxis) {
  OpTester test("OneHot", 11);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {2, 3, 10},
                          {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                           0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_float_float_float /*indices, output, depth*/) {
  OpTester test("OneHot", 11);
  test.AddInput<float>("indices", {2, 3}, {1., 9., 8., 2., 4., 6.});
  test.AddInput<float>("depth", {1}, {10.});
  test.AddInput<float>("values", {2}, {0., 1.});
  test.AddOutput<float>("output", {2, 3, 10},
                        {0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                         0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                         0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 1., 0., 0., 0.});
  // TRT EP segmentation fault in A100
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(OneHotOpTest, DefaultAxis_int64_int32_float /*indices, output, depth*/) {
  OpTester test("OneHot", 11);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<float>("depth", {1}, {10.});
  test.AddInput<int32_t>("values", {2}, {0, 1});
  test.AddOutput<int32_t>("output", {2, 3, 10},
                          {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                           0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(OneHotOpTest, DefaultAxis_int64_float_int64 /*indices, output, depth*/) {
  OpTester test("OneHot", 11);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<float>("values", {2}, {0, 1});
  test.AddOutput<float>("output", {2, 3, 10},
                        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                         0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                         0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_int32_float_float /*indices, output, depth*/) {
  OpTester test("OneHot", 11);
  test.AddInput<int32_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<float>("depth", {1}, {10.0f});
  test.AddInput<float>("values", {2}, {0.0f, 1.0f});
  test.AddOutput<float>("output", {2, 3, 10},
                        {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(OneHotOpTest, DefaultAxis_int32_float_int32 /*indices, output, depth*/) {
  OpTester test("OneHot", 11);
  test.AddInput<int32_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int32_t>("depth", {1}, {10});
  test.AddInput<float>("values", {2}, {0.0f, 1.0f});
  test.AddOutput<float>("output", {2, 3, 10},
                        {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});
  test.Run();
}

TEST(OneHotOpTest, Axis_0) {
  OpTester test("OneHot", 11);
  int64_t axis = 0;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {10, 2, 3},
                          {0, 0, 0, 0, 0, 0,
                           1, 0, 0, 0, 0, 0,
                           0, 0, 0, 1, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 1, 0, 0, 0,
                           0, 1, 0, 0, 0, 0});
  test.Run();
}

TEST(OneHotOpTest, Axis_1) {
  OpTester test("OneHot", 11);
  int64_t axis = 1;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {2, 10, 3},
                          {0, 0, 0,
                           1, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 1,
                           0, 1, 0,
                           0, 0, 0,
                           0, 0, 0,
                           1, 0, 0,
                           0, 0, 0,
                           0, 1, 0,
                           0, 0, 0,
                           0, 0, 1,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0});
  test.Run();
}

TEST(OneHotOpTest, Axis_2) {
  OpTester test("OneHot", 11);
  int64_t axis = 2;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {2, 3, 10},
                          {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                           0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
  test.Run();
}

TEST(OneHotOpTest, Axis_Negative_NonDefault) {
  OpTester test("OneHot", 11);
  int64_t axis = -3;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {10, 2, 3},
                          {0, 0, 0, 0, 0, 0,
                           1, 0, 0, 0, 0, 0,
                           0, 0, 0, 1, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 1, 0, 0, 0,
                           0, 1, 0, 0, 0, 0});
  test.Run();
}

TEST(OneHotOpTest, FloatInt64) {
  OpTester test("OneHot", 11);
  test.AddInput<float>("indices", {2, 3}, {1.f, 9.f, 8.f, 2.f, 4.f, 6.f});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {2, 3, 10},
                          {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                           0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_NonZeroOffValue) {
  OpTester test("OneHot", 11);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {2, 3});
  test.AddOutput<int64_t>("output", {2, 3, 10},
                          {2, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                           2, 2, 2, 2, 2, 2, 2, 2, 3, 2,
                           2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 3, 2, 2, 2});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_float_float_float_NonZeroOffValue /*indices, output, depth*/) {
  OpTester test("OneHot", 11);
  test.AddInput<float>("indices", {2, 3}, {1., 9., 8., 2., 4., 6.});
  test.AddInput<float>("depth", {1}, {10.});
  test.AddInput<float>("values", {2}, {2., 3.});
  test.AddOutput<float>("output", {2, 3, 10},
                        {2., 3., 2., 2., 2., 2., 2., 2., 2., 2.,
                         2., 2., 2., 2., 2., 2., 2., 2., 2., 3.,
                         2., 2., 2., 2., 2., 2., 2., 2., 3., 2.,
                         2., 2., 3., 2., 2., 2., 2., 2., 2., 2.,
                         2., 2., 2., 2., 3., 2., 2., 2., 2., 2.,
                         2., 2., 2., 2., 2., 2., 3., 2., 2., 2.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(OneHotOpTest, DefaultAxis_int64_int32_float_NonZeroOffValue /*indices, output, depth*/) {
  OpTester test("OneHot", 11);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<float>("depth", {1}, {10.});
  test.AddInput<int32_t>("values", {2}, {2, 3});
  test.AddOutput<int32_t>("output", {2, 3, 10},
                          {2, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                           2, 2, 2, 2, 2, 2, 2, 2, 3, 2,
                           2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 3, 2, 2, 2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(OneHotOpTest, DefaultAxis_int64_float_int64_NonZeroOffValue /*indices, output, depth*/) {
  OpTester test("OneHot", 11);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<float>("values", {2}, {2, 3});
  test.AddOutput<float>("output", {2, 3, 10},
                        {2, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                         2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                         2, 2, 2, 2, 2, 2, 2, 2, 3, 2,
                         2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
                         2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
                         2, 2, 2, 2, 2, 2, 3, 2, 2, 2});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_int32_float_float_NonZeroOffValue /*indices, output, depth*/) {
  OpTester test("OneHot", 11);
  test.AddInput<int32_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<float>("depth", {1}, {10.0f});
  test.AddInput<float>("values", {2}, {2.0f, 3.0f});
  test.AddOutput<float>("output", {2, 3, 10},
                        {2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                         2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f,
                         2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f,
                         2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                         2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                         2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(OneHotOpTest, DefaultAxis_int32_float_int32_NonZeroOffValue /*indices, output, depth*/) {
  OpTester test("OneHot", 11);
  test.AddInput<int32_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int32_t>("depth", {1}, {10});
  test.AddInput<float>("values", {2}, {2.0f, 3.0f});
  test.AddOutput<float>("output", {2, 3, 10},
                        {2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                         2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f,
                         2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f,
                         2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                         2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                         2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f});
  test.Run();
}

TEST(OneHotOpTest, Axis_0_NonZeroOffValue) {
  OpTester test("OneHot", 11);
  int64_t axis = 0;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {2, 3});
  test.AddOutput<int64_t>("output", {10, 2, 3},
                          {2, 2, 2, 2, 2, 2,
                           3, 2, 2, 2, 2, 2,
                           2, 2, 2, 3, 2, 2,
                           2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 3, 2,
                           2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 3,
                           2, 2, 2, 2, 2, 2,
                           2, 2, 3, 2, 2, 2,
                           2, 3, 2, 2, 2, 2});
  test.Run();
}

TEST(OneHotOpTest, Axis_1_NonZeroOffValue) {
  OpTester test("OneHot", 11);
  int64_t axis = 1;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {2, 3});
  test.AddOutput<int64_t>("output", {2, 10, 3},
                          {2, 2, 2,
                           3, 2, 2,
                           2, 2, 2,
                           2, 2, 2,
                           2, 2, 2,
                           2, 2, 2,
                           2, 2, 2,
                           2, 2, 2,
                           2, 2, 3,
                           2, 3, 2,
                           2, 2, 2,
                           2, 2, 2,
                           3, 2, 2,
                           2, 2, 2,
                           2, 3, 2,
                           2, 2, 2,
                           2, 2, 3,
                           2, 2, 2,
                           2, 2, 2,
                           2, 2, 2});
  test.Run();
}

TEST(OneHotOpTest, Axis_2_NonZeroOffValue) {
  OpTester test("OneHot", 11);
  int64_t axis = 2;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {2, 3});
  test.AddOutput<int64_t>("output", {2, 3, 10},
                          {2, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                           2, 2, 2, 2, 2, 2, 2, 2, 3, 2,
                           2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 3, 2, 2, 2});
  test.Run();
}

TEST(OneHotOpTest, Axis_Negative_NonDefault_NonZeroOffValue) {
  OpTester test("OneHot", 11);
  int64_t axis = -3;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {2, 3});
  test.AddOutput<int64_t>("output", {10, 2, 3},
                          {2, 2, 2, 2, 2, 2,
                           3, 2, 2, 2, 2, 2,
                           2, 2, 2, 3, 2, 2,
                           2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 3, 2,
                           2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 3,
                           2, 2, 2, 2, 2, 2,
                           2, 2, 3, 2, 2, 2,
                           2, 3, 2, 2, 2, 2});
  test.Run();
}

TEST(OneHotOpTest, FloatInt64_NonZeroOffValue) {
  OpTester test("OneHot", 11);
  test.AddInput<float>("indices", {2, 3}, {1.f, 9.f, 8.f, 2.f, 4.f, 6.f});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {2, 3});
  test.AddOutput<int64_t>("output", {2, 3, 10},
                          {2, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                           2, 2, 2, 2, 2, 2, 2, 2, 3, 2,
                           2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 3, 2, 2, 2});
  test.Run();
}

TEST(OneHotOpTest, FloatString) {
  OpTester test("OneHot", 11);
  test.AddInput<float>("indices", {2, 3}, {1.f, 9.f, 8.f, 2.f, 4.f, 6.f});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<string>("values", {2}, {"off", "on"});
  test.AddOutput<string>("output", {2, 3, 10},
                         {"off", "on", "off", "off", "off", "off", "off", "off", "off", "off",
                          "off", "off", "off", "off", "off", "off", "off", "off", "off", "on",
                          "off", "off", "off", "off", "off", "off", "off", "off", "on", "off",
                          "off", "off", "on", "off", "off", "off", "off", "off", "off", "off",
                          "off", "off", "off", "off", "on", "off", "off", "off", "off", "off",
                          "off", "off", "off", "off", "off", "off", "on", "off", "off", "off"});
  test.Run();
}

TEST(OneHotOpTest, Axis_Negative_NegIndex_NonDefault) {
  OpTester test("OneHot", 11);
  int64_t axis = -3;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, -1, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {10, 2, 3},
                          {0, 0, 0, 0, 0, 0,
                           1, 0, 0, 0, 0, 0,
                           0, 0, 0, 1, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 1, 0, 0, 0,
                           0, 1, 0, 0, 0, 0});
  test.Run();
}

TEST(OneHotOpTest, Axis_Negative_NegIndex_NonDefault_NonZeroOffValue) {
  OpTester test("OneHot", 11);
  int64_t axis = -3;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, -1, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {2, 3});
  test.AddOutput<int64_t>("output", {10, 2, 3},
                          {2, 2, 2, 2, 2, 2,
                           3, 2, 2, 2, 2, 2,
                           2, 2, 2, 3, 2, 2,
                           2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 3, 2,
                           2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 3,
                           2, 2, 2, 2, 2, 2,
                           2, 2, 3, 2, 2, 2,
                           2, 3, 2, 2, 2, 2});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_IndicesOutOfRange) {
  OpTester test("OneHot", 11);
  test.AddInput<int64_t>("indices", {2, 3}, {1, -1, 8, 13, 4, -12});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {2, 3, 10},
                          {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_IndicesOutOfRange_NonZeroOffValue) {
  OpTester test("OneHot", 11);
  test.AddInput<int64_t>("indices", {2, 3}, {1, -1, 8, 13, 4, -12});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {2, 3});
  test.AddOutput<int64_t>("output", {2, 3, 10},
                          {2, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                           2, 2, 2, 2, 2, 2, 2, 2, 3, 2,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
  test.Run();
}

TEST(OneHotOpTest, DimWithZero) {
  OpTester test("OneHot", 11);
  int64_t axis = 0;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 0}, {});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {10, 2, 0}, {});
  test.Run();
}

#ifdef USE_CUDA

TEST(OneHotOpTest, DefaultAxis_int64_MLFloat16_int64 /*indices, output, depth*/) {
  OpTester test("OneHot", 11);

  std::vector<float> values{0.0f, 1.0f};
  std::vector<MLFloat16> fp16_values(values.size());
  ConvertFloatToMLFloat16(values.data(), fp16_values.data(), static_cast<int>(values.size()));

  std::vector<float> output{0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<MLFloat16> fp16_output(output.size());
  ConvertFloatToMLFloat16(output.data(), fp16_output.data(), static_cast<int>(output.size()));

  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<MLFloat16>("values", {2}, fp16_values);
  test.AddOutput<MLFloat16>("output", {2, 3, 10}, fp16_output);

  // exclude CPU Execution Provider as MLFloat16 is not supported in CPU
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
}

TEST(OneHotOpTest, DefaultAxis_int32_MLFloat16_int32 /*indices, output, depth*/) {
  OpTester test("OneHot", 11);

  std::vector<float> values{0.0f, 1.0f};
  std::vector<MLFloat16> fp16_values(values.size());
  ConvertFloatToMLFloat16(values.data(), fp16_values.data(), static_cast<int>(values.size()));

  std::vector<float> output{0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<MLFloat16> fp16_output(output.size());
  ConvertFloatToMLFloat16(output.data(), fp16_output.data(), static_cast<int>(output.size()));

  test.AddInput<int32_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int32_t>("depth", {1}, {10});
  test.AddInput<MLFloat16>("values", {2}, fp16_values);
  test.AddOutput<MLFloat16>("output", {2, 3, 10}, fp16_output);

  // exclude CPU Execution Provider as MLFloat16 is not supported in CPU
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
}

TEST(OneHotOpTest, DefaultAxis_int64_MLFloat16_int64_NonZeroOffValue /*indices, output, depth*/) {
  OpTester test("OneHot", 11);

  std::vector<float> values{2.0f, 3.0f};
  std::vector<MLFloat16> fp16_values(values.size());
  ConvertFloatToMLFloat16(values.data(), fp16_values.data(), static_cast<int>(values.size()));

  std::vector<float> output{2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                            2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f,
                            2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f,
                            2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                            2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                            2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f};
  std::vector<MLFloat16> fp16_output(output.size());
  ConvertFloatToMLFloat16(output.data(), fp16_output.data(), static_cast<int>(output.size()));

  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<MLFloat16>("values", {2}, fp16_values);
  test.AddOutput<MLFloat16>("output", {2, 3, 10}, fp16_output);

  // exclude CPU Execution Provider as MLFloat16 is not supported in CPU
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
}

TEST(OneHotOpTest, DefaultAxis_int32_MLFloat16_int32_NonZeroOffValue /*indices, output, depth*/) {
  OpTester test("OneHot", 11);

  std::vector<float> values{2.0f, 3.0f};
  std::vector<MLFloat16> fp16_values(values.size());
  ConvertFloatToMLFloat16(values.data(), fp16_values.data(), static_cast<int>(values.size()));

  std::vector<float> output{2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                            2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f,
                            2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f,
                            2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                            2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                            2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f};
  std::vector<MLFloat16> fp16_output(output.size());
  ConvertFloatToMLFloat16(output.data(), fp16_output.data(), static_cast<int>(output.size()));

  test.AddInput<int32_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int32_t>("depth", {1}, {10});
  test.AddInput<MLFloat16>("values", {2}, fp16_values);
  test.AddOutput<MLFloat16>("output", {2, 3, 10}, fp16_output);

  // exclude CPU Execution Provider as MLFloat16 is not supported in CPU
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
}

#endif

}  // namespace test
}  // namespace onnxruntime
