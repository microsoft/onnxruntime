// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
void RunTypedTest()
{
  // int32_t indices - axis 0
  OpTester test1("GatherElements", 11);

  test1.AddAttribute<int64_t>("axis", 0LL);
  test1.AddInput<T>("data", {2, 3},
                       {0, 1, 2, 3, 4, 5});
  test1.AddInput<int32_t>("indices", {1, 2}, {0, 1});
  test1.AddOutput<T>("output", {1, 2},
                        {0, 4});
  test1.Run();

  // int32_t indices - axis 1
  OpTester test2("GatherElements", 11);
  test2.AddAttribute<int64_t>("axis", 1LL);
  test2.AddInput<T>("data", {2, 2},
                       {1, 2,
                        3, 4});
  test2.AddInput<int32_t>("indices", {2, 2},
                         {0, 0,
                          1, 0});
  test2.AddOutput<T>("output", {2, 2},
                        {1, 1,
                         4, 3});
  test2.Run();

  
  // int64_t indices - axis 1
  OpTester test3("GatherElements", 11);
  test3.AddAttribute<int64_t>("axis", 1LL);
  test3.AddInput<T>("data", {2, 2},
                       {1, 2,
                        3, 4});
  test3.AddInput<int64_t>("indices", {2, 2},
                         {0, 0,
                          1, 0});
  test3.AddOutput<T>("output", {2, 2},
                        {1, 1,
                         4, 3});
  test3.Run();
}

template <>
void RunTypedTest<std::string>() {

  // non-inner dimension 
  OpTester test1("GatherElements", 11);
  test1.AddAttribute<int64_t>("axis", 0LL);
  test1.AddInput<std::string>("data", {2, 3},
                             {"a", "b", "c", "d", "e", "f"});
  test1.AddInput<int32_t>("indices", {1, 2}, {0, 1});
  test1.AddOutput<std::string>("output", {1, 2},
                              {"a", "e"});
  test1.Run();

  // inner-dimension
  OpTester test2("GatherElements", 11);
  test2.AddAttribute<int64_t>("axis", 1LL);
  test2.AddInput<std::string>("data", {2, 2},
                             {"a", "b",
                              "c", "d"});
  test2.AddInput<int32_t>("indices", {2, 2},
                         {0, 0,
                          1, 0});
  test2.AddOutput<std::string>("output", {2, 2},
                              {"a", "a",
                               "d", "c"});
  test2.Run();
}

TEST(GatherElementsOpTest, int8_t) {
  RunTypedTest<int8_t>();
}

TEST(GatherElementsOpTest, int16_t) {
  RunTypedTest<int16_t>();
}

TEST(GatherElementsOpTest, int32_t) {
  RunTypedTest<int32_t>();
}

TEST(GatherElementsOpTest, int64_t) {
  RunTypedTest<int64_t>();
}

TEST(GatherElementsOpTest, string) {
  RunTypedTest<std::string>();
}

}  // namespace test
}  // namespace onnxruntime
