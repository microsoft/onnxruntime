// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

using namespace std;

namespace onnxruntime {
namespace test {

TEST(OneHotOpTest, DefaultAxis) {
  OpTester test("OneHot", 9);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {2, 3, 10}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_float_float_float /*indices, output, depth*/) {
  OpTester test("OneHot", 9);
  test.AddInput<float>("indices", {2, 3}, {1., 9., 8., 2., 4., 6.});
  test.AddInput<float>("depth", {1}, {10.});
  test.AddInput<float>("values", {2}, {0., 1.});
  test.AddOutput<float>("output", {2, 3, 10}, {0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                                 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                                                 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                                                 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                                                 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_int64_int32_float /*indices, output, depth*/) {
  OpTester test("OneHot", 9);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<float>("depth", {1}, {10.});
  test.AddInput<int32_t>("values", {2}, {0, 1});
  test.AddOutput<int32_t>("output", {2, 3, 10}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_int64_float_int64 /*indices, output, depth*/) {
  OpTester test("OneHot", 9);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<float>("values", {2}, {0, 1});
  test.AddOutput<float>("output", {2, 3, 10}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_int32_float_float /*indices, output, depth*/) {
  OpTester test("OneHot", 9);
  test.AddInput<int32_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<float>("depth", {1}, {10.0f});
  test.AddInput<float>("values", {2}, {0.0f, 1.0f});
  test.AddOutput<float>("output", {2, 3, 10}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                               0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});
  test.Run();
}

TEST(OneHotOpTest, DefaultAxis_int32_float_int32 /*indices, output, depth*/) {
  OpTester test("OneHot", 9);
  test.AddInput<int32_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int32_t>("depth", {1}, {10});
  test.AddInput<float>("values", {2}, {0.0f, 1.0f});
  test.AddOutput<float>("output", {2, 3, 10}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                               0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});
  test.Run();
}

TEST(OneHotOpTest, Axis_0) {
  OpTester test("OneHot", 9);
  int64_t axis = 0;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {10, 2, 3}, { 0, 0, 0, 0, 0, 0,
                                                  1, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 1, 0, 0,
                                                  0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 1, 0,
                                                  0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 1,
                                                  0, 0, 0, 0, 0, 0,
                                                  0, 0, 1, 0, 0, 0,
                                                  0, 1, 0, 0, 0, 0,});
  test.Run();
}

TEST(OneHotOpTest, Axis_1) {
  OpTester test("OneHot", 9);
  int64_t axis = 1;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {2, 10, 3}, { 0, 0, 0,
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
                                                  0, 0, 0,});
  test.Run();
}

TEST(OneHotOpTest, Axis_2) {
  OpTester test("OneHot", 9);
  int64_t axis = 2;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {2, 3, 10}, { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                                  0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 1, 0, 0, 0,});
  test.Run();
}

TEST(OneHotOpTest, Axis_Negative_NonDefault) {
  OpTester test("OneHot", 9);
  int64_t axis = -3;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1, 9, 8, 2, 4, 6});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {10, 2, 3}, { 0, 0, 0, 0, 0, 0,
                                                  1, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 1, 0, 0,
                                                  0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 1, 0,
                                                  0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 1,
                                                  0, 0, 0, 0, 0, 0,
                                                  0, 0, 1, 0, 0, 0,
                                                  0, 1, 0, 0, 0, 0,});
  test.Run();
}

TEST(OneHotOpTest, FloatInt64) {
  OpTester test("OneHot", 9);
  test.AddInput<float>("indices", {2, 3}, {1.f, 9.f, 8.f, 2.f, 4.f, 6.f});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<int64_t>("values", {2}, {0, 1});
  test.AddOutput<int64_t>("output", {2, 3, 10}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,});
  test.Run();
}

TEST(OneHotOpTest, FloatString) {
  OpTester test("OneHot", 9);
  test.AddInput<float>("indices", {2, 3}, {1.f, 9.f, 8.f, 2.f, 4.f, 6.f});
  test.AddInput<int64_t>("depth", {1}, {10});
  test.AddInput<string>("values", {2}, {"off", "on"});
  test.AddOutput<string>("output", {2, 3, 10}, {"off", "on", "off", "off", "off", "off", "off", "off", "off", "off",
                                                "off", "off", "off", "off", "off", "off", "off", "off", "off", "on",
                                                "off", "off", "off", "off", "off", "off", "off", "off", "on", "off",
                                                "off", "off", "on", "off", "off", "off", "off", "off", "off", "off",
                                                "off", "off", "off", "off", "on", "off", "off", "off", "off", "off",
                                                "off", "off", "off", "off", "off", "off", "on", "off", "off", "off",});
  test.Run();
}
}
}  // namespace onnxruntime
