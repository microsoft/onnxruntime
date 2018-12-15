// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

using namespace std;

namespace onnxruntime {
namespace test {

TEST(OneHotOpTest, Default) {
  OpTester test("OneHot", 9);
  test.AddInput<int64_t>("indices", {2, 3}, {1,9,8,2,4,6});
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

TEST(OneHotOpTest, Axis_0) {
  OpTester test("OneHot", 9);
  int64_t axis = 0;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("indices", {2, 3}, {1,9,8,2,4,6});
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
  test.AddInput<int64_t>("indices", {2, 3}, {1,9,8,2,4,6});
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
  test.AddInput<int64_t>("indices", {2, 3}, {1,9,8,2,4,6});
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
}
}  // namespace onnxruntime
