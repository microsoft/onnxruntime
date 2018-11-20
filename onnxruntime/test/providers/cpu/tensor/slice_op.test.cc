// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(SliceTest, Slice1D) {
  OpTester test("Slice");

  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("starts", std::vector<int64_t>{2});
  test.AddAttribute("ends", std::vector<int64_t>{4});

  test.AddInput<float>("data", {6}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddOutput<float>("output", {2}, {2.0f, 3.0f});
  test.Run();
}

TEST(SliceTest, Slice1D_Perf) {
  OpTester test("Slice");

  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("starts", std::vector<int64_t>{2});
  test.AddAttribute("ends", std::vector<int64_t>{502});

  std::vector<float> input(1000, 2.0f);
  std::vector<float> output(500, 2.0f);
  test.AddInput<float>("data", {1000}, input);
  test.AddOutput<float>("output", {500}, output);
  test.Run();
}

TEST(SliceTest, Slice2D_OutOfBounds) {
  OpTester test("Slice");

  test.AddAttribute("axes", std::vector<int64_t>{0, 1});
  test.AddAttribute("starts", std::vector<int64_t>{0, 1000});
  test.AddAttribute("ends", std::vector<int64_t>{10, 1000});

  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddOutput<float>("output", {2, 0}, {});
  test.Run();
}

TEST(SliceTest, Slice2D_OneAxis) {
  OpTester test("Slice");

  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("starts", std::vector<int64_t>{1});
  test.AddAttribute("ends", std::vector<int64_t>{3});

  test.AddInput<float>("data", {6, 4},
                       {00.0f, 01.0f, 02.0f, 03.0f,
                        10.0f, 11.0f, 12.0f, 13.0f,
                        20.0f, 21.0f, 22.0f, 23.0f,
                        30.0f, 31.0f, 32.0f, 33.0f,
                        40.0f, 41.0f, 42.0f, 43.0f,
                        50.0f, 51.0f, 52.0f, 53.0f});
  test.AddOutput<float>("output", {2, 4},
                        {10.0f, 11.0f, 12.0f, 13.0f,
                         20.0f, 21.0f, 22.0f, 23.0f});
  test.Run();
}

TEST(SliceTest, Slice2D_TwoAxes) {
  OpTester test("Slice");

  test.AddAttribute("axes", std::vector<int64_t>{1, 0});
  test.AddAttribute("starts", std::vector<int64_t>{2, 3});
  test.AddAttribute("ends", std::vector<int64_t>{1000, -1});

  test.AddInput<float>("data", {6, 4},
                       {00.0f, 01.0f, 02.0f, 03.0f,
                        10.0f, 11.0f, 12.0f, 13.0f,
                        20.0f, 21.0f, 22.0f, 23.0f,
                        30.0f, 31.0f, 32.0f, 33.0f,
                        40.0f, 41.0f, 42.0f, 43.0f,
                        50.0f, 51.0f, 52.0f, 53.0f});
  test.AddOutput<float>("output", {2, 2},
                        {32.0f, 33.0f,
                         42.0f, 43.0f});
  test.Run();
}

TEST(SliceTest, Slice2D_TwoAxesEque) {
  OpTester test("Slice");

  test.AddAttribute("axes", std::vector<int64_t>{1, 0});
  test.AddAttribute("starts", std::vector<int64_t>{2, 3});
  test.AddAttribute("ends", std::vector<int64_t>{1000, 3});

  test.AddInput<float>("data", {6, 4},
                       {00.0f, 01.0f, 02.0f, 03.0f,
                        10.0f, 11.0f, 12.0f, 13.0f,
                        20.0f, 21.0f, 22.0f, 23.0f,
                        30.0f, 31.0f, 32.0f, 33.0f,
                        40.0f, 41.0f, 42.0f, 43.0f,
                        50.0f, 51.0f, 52.0f, 53.0f});
  test.AddOutput<float>("output", {0, 2},
                        {});
  test.Run();
}

TEST(SliceTest, Slice3D) {
  OpTester test("Slice");

  test.AddAttribute("starts", std::vector<int64_t>{0, 1, 1});
  test.AddAttribute("ends", std::vector<int64_t>{1000, 1000, 1000});

  test.AddInput<float>("data", {3, 3, 3},
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f,

                        211.0f, 212.0f, 213.0f,
                        221.0f, 222.0f, 223.0f,
                        231.0f, 232.0f, 233.0f,

                        311.0f, 312.0f, 313.0f,
                        321.0f, 322.0f, 323.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("output", {3, 2, 2},
                        {122.0f, 123.0f,
                         132.0f, 133.0f,

                         222.0f, 223.0f,
                         232.0f, 233.0f,

                         322.0f, 323.0f,
                         332.0f, 333.0f});
  test.Run();
}

}  // namespace Test
}  // namespace onnxruntime
