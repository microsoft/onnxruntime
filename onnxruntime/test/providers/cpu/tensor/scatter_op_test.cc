// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void scatter_without_axis_tests(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);

  std::vector<float> input;
  input.resize(3 * 3);
  std::fill(input.begin(), input.end(), .0f);
  test.AddInput<float>("data", {3, 3}, input);

  test.AddInput<int64_t>("indices", {2, 3},
                         {1, 0, 2,
                          0, 2, 1});

  test.AddInput<float>("updates", {2, 3},
                       {1.0f, 1.1f, 1.2f,
                        2.0f, 2.1f, 2.2f});

  test.AddOutput<float>("y", {3, 3},
                        {2.0f, 1.1f, 0.0f,
                         1.0f, 0.0f, 2.2f,
                         0.0f, 2.1f, 1.2f});
  test.Run();
}

TEST(Scatter, WithoutAxis) {
  scatter_without_axis_tests("Scatter", 9);
  scatter_without_axis_tests("ScatterElements", 11);
}

static void scatter_with_axis_tests(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 1);

  test.AddInput<float>("data", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<int64_t>("indices", {1, 2}, {1, 3});
  test.AddInput<float>("updates", {1, 2}, {1.1f, 2.1f});
  test.AddOutput<float>("y", {1, 5}, {1.0f, 1.1f, 3.0f, 2.1f, 5.0f});
  test.Run();
}

TEST(Scatter, WithAxis) {
  scatter_with_axis_tests("Scatter", 9);
  scatter_with_axis_tests("ScatterElements", 11);
}

static void scatter_three_dim_with_axis_0(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 0);

  test.AddInput<int64_t>("data", {1, 3, 3},
                         {1, 2, 3,
                          4, 5, 6,
                          7, 8, 9});
  // Because axis 0 is only 1 dimension it should be all zeros
  test.AddInput<int64_t>("indices", {1, 3, 3},
                         {0, 0, 0,
                          0, 0, 0,
                          0, 0, 0});
  test.AddInput<int64_t>("updates", {1, 3, 3},
                         {11, 12, 13,
                          14, 15, 16,
                          17, 18, 19});
  test.AddOutput<int64_t>("y", {1, 3, 3},
                          {11, 12, 13,
                           14, 15, 16,
                           17, 18, 19});
  test.Run();
}

TEST(Scatter, ThreeDimsWithAxis_0) {
  scatter_three_dim_with_axis_0("Scatter", 9);
  scatter_three_dim_with_axis_0("ScatterElements", 11);
}

static void scatter_three_dim_with_axis_negative_2(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", -2);

  test.AddInput<int64_t>("data", {2, 2, 2},
                         {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddInput<int64_t>("indices", {2, 1, 2},
                         {0, 1, 1, 0});
  test.AddInput<int64_t>("updates", {2, 1, 2},
                         {11, 12, 13, 14});
  test.AddOutput<int64_t>("y", {2, 2, 2},
                          {11, 2, 3, 12, 5, 14, 13, 8});
  test.Run();
}

TEST(Scatter, ThreeDimsWithAxisNegative_2) {
  scatter_three_dim_with_axis_negative_2("Scatter", 9);
  scatter_three_dim_with_axis_negative_2("ScatterElements", 11);
}

static void scatter_three_dim_with_axis_2(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 2);

  test.AddInput<int64_t>("data", {1, 3, 3},
                         {1, 2, 3,
                          4, 5, 6,
                          7, 8, 9});

  test.AddInput<int64_t>("indices", {1, 3, 3},
                         {2, 1, 0,
                          2, 1, 0,
                          2, 1, 0});

  test.AddInput<int64_t>("updates", {1, 3, 3},
                         {11, 12, 13,
                          14, 15, 16,
                          17, 18, 19});
  test.AddOutput<int64_t>("y", {1, 3, 3},
                          {13, 12, 11,
                           16, 15, 14,
                           19, 18, 17});
  test.Run();
}

TEST(Scatter, ThreeDimsWithAxis_2) {
  scatter_three_dim_with_axis_2("Scatter", 9);
  scatter_three_dim_with_axis_2("ScatterElements", 11);
}

static void scatter_string(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 1);

  test.AddInput<std::string>("data", {1, 5}, {"1.0f", "2.0f", "3.0f", "4.0f", "5.0f"});
  test.AddInput<int64_t>("indices", {1, 2}, {1, 3});
  test.AddInput<std::string>("updates", {1, 2}, {"1.1f", "2.1f"});
  test.AddOutput<std::string>("y", {1, 5}, {"1.0f", "1.1f", "3.0f", "2.1f", "5.0f"});
  test.Run();
}

TEST(Scatter, String) {
  scatter_string("Scatter", 9);
  scatter_string("ScatterElements", 11);
}

static void scatter_negative_axis(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", -1);

  test.AddInput<float>("data", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<int64_t>("indices", {1, 2}, {1, 3});
  test.AddInput<float>("updates", {1, 2}, {1.1f, 2.1f});
  test.AddOutput<float>("y", {1, 5}, {1.0f, 1.1f, 3.0f, 2.1f, 5.0f});
  #if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M) //TBD temporarily disabling for openvino
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  #else
    test.Run();
  #endif
}

TEST(Scatter, NegativeAxis) {
  scatter_negative_axis("Scatter", 9);
  scatter_negative_axis("ScatterElements", 11);
}

static void scatter_indices_updates_dont_match(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 1);

  test.AddInput<float>("data", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<int64_t>("indices", {1, 3}, {1, 3, 3});
  test.AddInput<float>("updates", {1, 2}, {1.1f, 2.1f});
  test.AddOutput<float>("y", {1, 5}, {1.0f, 1.1f, 3.0f, 2.1f, 5.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "Indices vs updates dimensions differs at position=1 3 vs 2");
}

TEST(Scatter, IndicesUpdatesDontMatch) {
  scatter_indices_updates_dont_match("Scatter", 9);
  scatter_indices_updates_dont_match("ScatterElements", 11);
}

static void scatter_valid_index(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 0);

  test.AddInput<float>("data", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddInput<int64_t>("indices", {1, 1, 1}, {3});
  test.AddInput<float>("updates", {1, 1, 1}, {5.0f});
  test.AddOutput<float>("y", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f});
  test.Run();
}

TEST(Scatter, ValidAxis) {
  scatter_valid_index("Scatter", 9);
  scatter_valid_index("ScatterElements", 11);
}

static void scatter_invalid_index(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 0);

  test.AddInput<float>("data", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddInput<int64_t>("indices", {1, 1, 1}, {4});
  test.AddInput<float>("updates", {1, 1, 1}, {5.0f});
  test.AddOutput<float>("y", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "indices element out of data bounds, idx=4 must be within the inclusive range [-4,3]",
           {kCudaExecutionProvider});
}

TEST(Scatter, InvalidIndex) {
  scatter_invalid_index("Scatter", 9);
  scatter_invalid_index("ScatterElements", 11);
}

static void scatter_valid_negative_index(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 0);

  test.AddInput<float>("data", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddInput<int64_t>("indices", {1, 1, 1}, {-1});
  test.AddInput<float>("updates", {1, 1, 1}, {5.0f});
  test.AddOutput<float>("y", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f});
  #if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M) //TBD temporarily disabling for openvino
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  #else
    test.Run();
  #endif
}

TEST(Scatter, ValidNegativeIndex) {
  scatter_valid_negative_index("Scatter", 9);
  scatter_valid_negative_index("ScatterElements", 11);
}

static void scatter_bool_with_axis_tests(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 1);

  test.AddInput<bool>("data", {1, 5}, {false, false, false, true, false});
  test.AddInput<int64_t>("indices", {1, 2}, {1, 3});
  test.AddInput<bool>("updates", {1, 2}, {true, false});
  test.AddOutput<bool>("y", {1, 5}, {false, true, false, false, false});
  test.Run();
}

TEST(Scatter, BoolInputWithAxis) {
  scatter_bool_with_axis_tests("Scatter", 9);
  scatter_bool_with_axis_tests("ScatterElements", 11);
}

static void scatter_same_updates_tests(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);

  std::vector<float> input;
  input.resize(3 * 3);
  std::fill(input.begin(), input.end(), .0f);
  test.AddInput<float>("data", {3, 3}, input);

  test.AddInput<int64_t>("indices", {1, 2},
                         {1, 1}, /*is_initializer*/ true);

  test.AddInput<float>("updates", {1, 2},
                       {2.0f, 2.0f});

  test.AddOutput<float>("y", {3, 3},
                        {0.0f, 0.0f, 0.0f,
                         2.0f, 2.0f, 0.0f,
                         0.0f, 0.0f, 0.0f});
  test.Run();
}

TEST(Scatter, SameUpdateWithoutAxis) {
  scatter_same_updates_tests("Scatter", 9);
  scatter_same_updates_tests("ScatterElements", 11);
}

static void scatter_with_larger_indices_on_axis_tests(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 1);

  test.AddInput<float>("data", {1, 2}, {1.0f, 2.0f});
  test.AddInput<int64_t>("indices", {1, 4}, {0, 0, 0, 0});
  test.AddInput<float>("updates", {1, 4}, {3.0f, 3.0f, 3.0f, 3.0f});
  test.AddOutput<float>("y", {1, 2}, {3.0f, 2.0f});
  test.Run();
}

TEST(Scatter, LargerIndicesOnAxis) {
  scatter_with_larger_indices_on_axis_tests("Scatter", 9);
  scatter_with_larger_indices_on_axis_tests("ScatterElements", 11);
}

}  // namespace test
}  // namespace onnxruntime
