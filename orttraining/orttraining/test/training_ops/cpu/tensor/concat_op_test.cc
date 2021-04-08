// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(ConcatTrainingOpTest, Concat1D_int32_negative_axis) {
  OpTester test("ConcatTraining", 1, kMSDomain);
  test.AddAttribute("axis", int64_t{-1});

  test.AddInput<int32_t>("input1", {1}, {1});
  test.AddInput<int32_t>("input2", {2}, {2, 3});
  test.AddInput<int32_t>("input3", {4}, {4, 5, 6, 7});
  test.AddOutput<int32_t>("concat_result", {7}, {1, 2, 3, 4, 5, 6, 7});
  test.AddOutput<int64_t>("per_input_length", {3}, {1, 2, 4});
  test.Run();
}

TEST(ConcatTrainingOpTest, Concat2D_float_axis1) {
  OpTester test("ConcatTraining", 1, kMSDomain);
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{4, 1};
  test.AddInput<float>("input1", dims, {11.0f, 21.0f, 31.0f, 41.0f});
  test.AddInput<float>("input2", {4, 2}, {12.0f, 13.0f, 22.0f, 23.0f, 32.0f, 33.0f, 42.0f, 43.0f});
  test.AddInput<float>("input3", dims, {14.0f, 24.0f, 34.0f, 44.0f});
  test.AddOutput<float>("concat_result", {4, 4},
                        {11.0f, 12.0f, 13.0f, 14.0f,
                         21.0f, 22.0f, 23.0f, 24.0f,
                         31.0f, 32.0f, 33.0f, 34.0f,
                         41.0f, 42.0f, 43.0f, 44.0f});
  test.AddOutput<int64_t>("per_input_length", {3}, {1, 2, 1});
  test.Run();
}

TEST(ConcatTrainingOpTest, Concat3D_same_len) {
  OpTester test("ConcatTraining", 1, kMSDomain);
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<float>("input1", dims,
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f});
  test.AddInput<float>("input2", dims,
                       {9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f});
  test.AddOutput<float>("concat_result", {2, 4, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f,
                         9.0f, 10.0f,
                         11.0f, 12.0f,

                         5.0f, 6.0f,
                         7.0f, 8.0f,
                         13.0f, 14.0f,
                         15.0f, 16.0f});
  test.AddOutput<int64_t>("per_input_length", {2}, {2, 2});
  test.Run();
}

TEST(ConcatTrainingOpTest, Concat2D_optional_output1) {
  OpTester test("ConcatTraining", 1, kMSDomain);
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{4, 1};
  test.AddInput<float>("input1", dims, {11.0f, 21.0f, 31.0f, 41.0f});
  test.AddInput<float>("input2", {4, 2}, {12.0f, 13.0f, 22.0f, 23.0f, 32.0f, 33.0f, 42.0f, 43.0f});
  test.AddInput<float>("input3", dims, {14.0f, 24.0f, 34.0f, 44.0f});
  test.AddOutput<float>("concat_result", {4, 4},
                        {11.0f, 12.0f, 13.0f, 14.0f,
                         21.0f, 22.0f, 23.0f, 24.0f,
                         31.0f, 32.0f, 33.0f, 34.0f,
                         41.0f, 42.0f, 43.0f, 44.0f});
  test.AddMissingOptionalOutput<int64_t>();
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
