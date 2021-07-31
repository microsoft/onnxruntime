// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
void RunTypedTest() {
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

  // negative indices - axis 1
  OpTester test4("GatherElements", 11);
  test4.AddAttribute<int64_t>("axis", 1LL);
  test4.AddInput<T>("data", {2, 2},
                    {1, 2,
                     3, 4});
  test4.AddInput<int64_t>("indices", {2, 2},
                          {0, 0,
                           -1, -1});
  test4.AddOutput<T>("output", {2, 2},
                     {1, 1,
                      4, 4});
  // skip TensorRT because it doesn't support negative indices				  
  test4.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  // indices out of bounds
  OpTester test5("GatherElements", 11);
  test5.AddAttribute<int64_t>("axis", 1LL);
  test5.AddInput<T>("data", {2, 2},
                    {1, 2,
                     3, 4});
  test5.AddInput<int64_t>("indices", {2, 2},
                          {0, 0,
                           2, 2});
  test5.AddOutput<T>("output", {2, 2},
                     {1, 1,
                      4, 4});
  // skip nuphar, which will not throw error message but will ensure no out-of-bound access
  // skip cuda as the cuda kernel won't throw the error message
  // skip openvino which will not throw error message but will ensure no out-of-bound access
  // skip TensorRT because it doesn't support out of bounds indices
  test5.Run(OpTester::ExpectResult::kExpectFailure,
            "GatherElements op: Value in indices must be within bounds [-2 , 1]. Actual value is 2",
            {kNupharExecutionProvider, kCudaExecutionProvider, kRocmExecutionProvider, kOpenVINOExecutionProvider, kTensorrtExecutionProvider});

  // 3D input - axis 1
  OpTester test6("GatherElements", 11);
  test6.AddAttribute<int64_t>("axis", 1LL);
  test6.AddInput<T>("data", {2, 2, 2},
                    {1, 2,
                     3, 4,
                     5, 6,
                     7, 8});
  test6.AddInput<int64_t>("indices", {1, 2, 1},
                          {0, 1});
  test6.AddOutput<T>("output", {1, 2, 1}, {1, 3});
  test6.Run();

  // 3D input - axis 2
  OpTester test7("GatherElements", 11);
  test7.AddAttribute<int64_t>("axis", 2LL);
  test7.AddInput<T>("data", {2, 2, 2},
                    {1, 2,
                     3, 4,
                     5, 6,
                     7, 8});
  test7.AddInput<int64_t>("indices", {1, 2, 1},
                          {0, 1});
  test7.AddOutput<T>("output", {1, 2, 1}, {1, 4});
  test7.Run();

  // 2D input - axis 1
  OpTester test8("GatherElements", 11);
  test8.AddAttribute<int64_t>("axis", 1LL);
  test8.AddInput<T>("data", {3, 3},
                    {1, 2, 3,
                     4, 5, 6,
                     7, 8, 9});
  test8.AddInput<int64_t>("indices", {3, 2},
                          {1, 0, 0, 1, 0, 1});
  test8.AddOutput<T>("output", {3, 2}, {2, 1, 4, 5, 7, 8});
  test8.Run();

  // 2D input - axis 1
  OpTester test9("GatherElements", 11);
  test9.AddAttribute<int64_t>("axis", 0LL);
  test9.AddInput<T>("data", {3, 3},
                    {1, 2, 3,
                     4, 5, 6,
                     7, 8, 9});
  test9.AddInput<int64_t>("indices", {3, 2},
                          {1, 0, 0, 1, 0, 1});
  test9.AddOutput<T>("output", {3, 2}, {4, 2, 1, 5, 1, 5});
  test9.Run();

  // 1D input - axis 0
  OpTester test10("GatherElements", 11);
  test10.AddAttribute<int64_t>("axis", 0LL);
  test10.AddInput<T>("data", {3},
                     {1, 2, 3});
  test10.AddInput<int64_t>("indices", {2},
                           {1, 0});
  test10.AddOutput<T>("output", {2}, {2, 1});
  test10.Run();
}

template <>
void RunTypedTest<bool>() {
  // 3D input - axis 2
  OpTester test1("GatherElements", 11);
  test1.AddAttribute<int64_t>("axis", 2LL);
  test1.AddInput<bool>("data", {2, 2, 2},
                       {true, false,
                        true, false,
                        true, false,
                        true, false});
  test1.AddInput<int64_t>("indices", {1, 2, 1},
                          {0, 1});
  test1.AddOutput<bool>("output", {1, 2, 1}, {true, false});
  test1.Run();
}

template <>
void RunTypedTest<std::string>() {
  // int32_t indices - axis 0
  OpTester test1("GatherElements", 11);
  test1.AddAttribute<int64_t>("axis", 0LL);
  test1.AddInput<std::string>("data", {2, 3},
                              {"a", "b", "c", "d", "e", "f"});
  test1.AddInput<int32_t>("indices", {1, 2}, {0, 1});
  test1.AddOutput<std::string>("output", {1, 2},
                               {"a", "e"});
  test1.Run();

  // int32_t indices - axis 1
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

  // negative indices - axis 1
  OpTester test3("GatherElements", 11);
  test3.AddAttribute<int64_t>("axis", 1LL);
  test3.AddInput<std::string>("data", {2, 2},
                              {"a", "b",
                               "c", "d"});
  test3.AddInput<int32_t>("indices", {2, 2},
                          {0, 0,
                           -1, -1});
  test3.AddOutput<std::string>("output", {2, 2},
                               {"a", "a",
                                "d", "d"});
  test3.Run();

  // indices out of bounds
  OpTester test4("GatherElements", 11);
  test4.AddAttribute<int64_t>("axis", 1LL);
  test4.AddInput<std::string>("data", {2, 2},
                              {"a", "b",
                               "c", "d"});
  test4.AddInput<int32_t>("indices", {2, 2},
                          {0, 0,
                           -3, -3});
  test4.AddOutput<std::string>("output", {2, 2},
                               {"a", "a",
                                "d", "d"});
  // skip nuphar, which will not throw error message but will ensure no out-of-bound access
  // skip Openvino, which will not throw error message but will ensure no out-of-bound access
  test4.Run(OpTester::ExpectResult::kExpectFailure,
            "GatherElements op: Value in indices must be within bounds [-2 , 1]. Actual value is -3",
            {kNupharExecutionProvider, kOpenVINOExecutionProvider});

  // 3D input - axis 1
  OpTester test5("GatherElements", 11);
  test5.AddAttribute<int64_t>("axis", 1LL);
  test5.AddInput<std::string>("data", {2, 2, 2},
                              {"a", "b",
                               "c", "d",
                               "e", "f",
                               "g", "h"});
  test5.AddInput<int32_t>("indices", {1, 2, 1},
                          {0, 1});
  test5.AddOutput<std::string>("output", {1, 2, 1},
                               {"a", "c"});
  test5.Run();

  // 3D input - axis 2
  OpTester test6("GatherElements", 11);
  test6.AddAttribute<int64_t>("axis", 2LL);
  test6.AddInput<std::string>("data", {2, 2, 2},
                              {"a", "b",
                               "c", "d",
                               "e", "f",
                               "g", "h"});
  test6.AddInput<int32_t>("indices", {1, 2, 1},
                          {0, 1});
  test6.AddOutput<std::string>("output", {1, 2, 1},
                               {"a", "d"});
  test6.Run();

  // 2D input - axis 1
  OpTester test7("GatherElements", 11);
  test7.AddAttribute<int64_t>("axis", 1LL);
  test7.AddInput<std::string>("data", {3, 3},
                              {"a", "b", "c",
                               "d", "e", "f",
                               "g", "h", "i"});
  test7.AddInput<int64_t>("indices", {3, 2},
                          {1, 0, 0, 1, 0, 1});
  test7.AddOutput<std::string>("output", {3, 2}, {"b", "a", "d", "e", "g", "h"});
  test7.Run();

  // 2D input - axis 2
  OpTester test8("GatherElements", 11);
  test8.AddAttribute<int64_t>("axis", 0LL);
  test8.AddInput<std::string>("data", {3, 3},
                              {"a", "b", "c",
                               "d", "e", "f",
                               "g", "h", "i"});
  test8.AddInput<int64_t>("indices", {3, 2},
                          {1, 0, 0, 1, 0, 1});
  test8.AddOutput<std::string>("output", {3, 2}, {"d", "b", "a", "e", "a", "e"});
  test8.Run();
}

// Disable TensorRT due to missing int8 calibrator
#if !defined(USE_TENSORRT)
TEST(GatherElementsOpTest, int8_t) {
  RunTypedTest<int8_t>();
}
#endif

TEST(GatherElementsOpTest, int16_t) {
  RunTypedTest<int16_t>();
}

TEST(GatherElementsOpTest, int32_t) {
  RunTypedTest<int32_t>();
}

TEST(GatherElementsOpTest, int64_t) {
  RunTypedTest<int64_t>();
}

TEST(GatherElementsOpTest, uint8_t) {
  RunTypedTest<uint8_t>();
}

TEST(GatherElementsOpTest, uint16_t) {
  RunTypedTest<uint16_t>();
}

TEST(GatherElementsOpTest, uint32_t) {
  RunTypedTest<uint32_t>();
}

TEST(GatherElementsOpTest, uint64_t) {
  RunTypedTest<uint64_t>();
}

TEST(GatherElementsOpTest, float) {
  RunTypedTest<float>();
}

TEST(GatherElementsOpTest, double) {
  RunTypedTest<double>();
}

TEST(GatherElementsOpTest, bool) {
  RunTypedTest<bool>();
}

TEST(GatherElementsOpTest, string) {
  RunTypedTest<std::string>();
}

TEST(GatherElementsOpTest, BigIndices) {
  // int32_t indices - axis 0
  OpTester test1("GatherElements", 11);

  test1.AddAttribute<int64_t>("axis", 0LL);
  const int kNumIndices = 10 * 1000;  // must be >= kParallelizationThreshold in gather_elements.cc
  std::vector<float> input(2 * kNumIndices);
  std::iota(std::begin(input), std::end(input), 0.f);
  test1.AddInput<float>("data", {2, kNumIndices}, input);

  std::vector<int32_t> indices(kNumIndices, 0);
  std::vector<float> output(kNumIndices);
  std::iota(std::begin(output), std::end(output), 0.f);
  test1.AddInput<int32_t>("indices", {1, kNumIndices}, indices);
  test1.AddOutput<float>("output", {1, kNumIndices}, output);
  test1.Run();
}

}  // namespace test
}  // namespace onnxruntime
