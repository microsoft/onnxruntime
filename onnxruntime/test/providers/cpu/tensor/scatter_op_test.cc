// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

const int Scatter_ver = 9;

TEST(ScatterOpTest, WithoutAxis) {
  OpTester test("Scatter", Scatter_ver);

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

TEST(ScatterOpTest, WithAxis) {
  OpTester test("Scatter", Scatter_ver);
  test.AddAttribute<int64_t>("axis", 1);

  test.AddInput<float>("data", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<int64_t>("indices", {1, 2}, {1, 3});
  test.AddInput<float>("updates", {1, 2}, {1.1f, 2.1f});
  test.AddOutput<float>("y", {1, 5}, {1.0f, 1.1f, 3.0f, 2.1f, 5.0f});
  test.Run();
}

TEST(ScatterOpTest, WithAxisThreeDims) {
  OpTester test("Scatter", Scatter_ver);
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

TEST(ScatterOpTest, ThreeDimsWithAxisGE_1) {
  OpTester test("Scatter", Scatter_ver);
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

TEST(ScatterOpTest, WithAxisStrings) {
  OpTester test("Scatter", Scatter_ver);
  test.AddAttribute<int64_t>("axis", 1);

  test.AddInput<std::string>("data", {1, 5}, {"1.0f", "2.0f", "3.0f", "4.0f", "5.0f"});
  test.AddInput<int64_t>("indices", {1, 2}, {1, 3});
  test.AddInput<std::string>("updates", {1, 2}, {"1.1f", "2.1f"});
  test.AddOutput<std::string>("y", {1, 5}, {"1.0f", "1.1f", "3.0f", "2.1f", "5.0f"});
  test.Run();
}

TEST(ScatterOpTest, NegativeAxis) {
  OpTester test("Scatter", Scatter_ver);
  test.AddAttribute<int64_t>("axis", -1);

  test.AddInput<float>("data", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<int64_t>("indices", {1, 2}, {1, 3});
  test.AddInput<float>("updates", {1, 2}, {1.1f, 2.1f});
  test.AddOutput<float>("y", {1, 5}, {1.0f, 1.1f, 3.0f, 2.1f, 5.0f});
  test.Run();
}

TEST(ScatterOpTest, IndicesUpdatesDimsDonotMatch) {
  OpTester test("Scatter", Scatter_ver);
  test.AddAttribute<int64_t>("axis", 1);

  test.AddInput<float>("data", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<int64_t>("indices", {1, 3}, {1, 3, 3});
  test.AddInput<float>("updates", {1, 2}, {1.1f, 2.1f});
  test.AddOutput<float>("y", {1, 5}, {1.0f, 1.1f, 3.0f, 2.1f, 5.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "Indices vs updates dimensions differs at position=1 3 vs 2");
}

TEST(ScatterOpTest, ValidIndex) {
  OpTester test("Scatter", Scatter_ver);
  test.AddAttribute<int64_t>("axis", 0);

  test.AddInput<float>("data", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddInput<int64_t>("indices", {1, 1, 1}, {3});
  test.AddInput<float>("updates", {1, 1, 1}, {5.0f});
  test.AddOutput<float>("y", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f});
  test.Run();
}

TEST(ScatterOpTest, InvalidIndex) {
  OpTester test("Scatter", Scatter_ver);
  test.AddAttribute<int64_t>("axis", 0);

  test.AddInput<float>("data", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddInput<int64_t>("indices", {1, 1, 1}, {4});
  test.AddInput<float>("updates", {1, 1, 1}, {5.0f});
  test.AddOutput<float>("y", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "indices element out of data bounds, idx=4 data_dim=4");
}
}  // namespace test
}  // namespace onnxruntime
