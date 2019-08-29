// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Some of the tests can't run on TensorrtExecutionProvider because of unsupported data types.
// Those tests will fallback to other EPs

TEST(GatherElementsOpTest, Gather_float_axis0_int32_indices) {
  OpTester test("GatherElements", 11);
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<float>("data", {2, 3},
                       {0.0f, 0.1f, 0.2f, 1.0f, 1.1f, 1.2f});
  test.AddInput<int32_t>("indices", {1, 2}, {0, 1});
  test.AddOutput<float>("output", {1, 2},
                        {0.0f, 1.1f});
  test.Run();
}

TEST(GatherElementsOpTest, Gather_float_axis1_int32_indices) {
  OpTester test("GatherElements", 11);
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<float>("data", {2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int32_t>("indices", {2, 2},
                         {0, 0,
                          1, 0});
  test.AddOutput<float>("output", {2, 2},
                        {1.0f, 1.0f,
                         4.0f, 3.0f});
  test.Run();
}

TEST(GatherElementsOpTest, Gather_string_axis0_int32_indices) {
  OpTester test("GatherElements", 11);
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<std::string>("data", {2, 3},
                             {"a", "b", "c", "d", "e", "f"});
  test.AddInput<int32_t>("indices", {1, 2}, {0, 1});
  test.AddOutput<std::string>("output", {1, 2},
                              {"a", "e"});
  test.Run();
}

TEST(GatherElementsOpTest, Gather_string_axis1_int32_indices) {
  OpTester test("GatherElements", 11);
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<std::string>("data", {2, 2},
                             {"a", "b",
                              "c", "d"});
  test.AddInput<int32_t>("indices", {2, 2},
                         {0, 0,
                          1, 0});
  test.AddOutput<std::string>("output", {2, 2},
                              {"a", "a",
                               "d", "c"});
  test.Run();
}

TEST(GatherElementsOpTest, Gather_float_axis0_int64_indices) {
  OpTester test("GatherElements", 11);
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<float>("data", {2, 3},
                       {0.0f, 0.1f, 0.2f, 1.0f, 1.1f, 1.2f});
  test.AddInput<int64_t>("indices", {1, 2}, {0, 1});
  test.AddOutput<float>("output", {1, 2},
                        {0.0f, 1.1f});
  test.Run();
}

TEST(GatherElementsOpTest, Gather_float_axis1_int64_indices) {
  OpTester test("GatherElements", 11);
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<float>("data", {2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("indices", {2, 2},
                         {0, 0,
                          1, 0});
  test.AddOutput<float>("output", {2, 2},
                        {1.0f, 1.0f,
                         4.0f, 3.0f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
