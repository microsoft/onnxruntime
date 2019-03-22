// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(TensorOpTest, Tile1DWithZeroRepeatsFloatInput) {
  OpTester test("Tile");

  test.AddInput<float>("input", {3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("repeats", {1}, {0});
  test.AddOutput<float>("output", {0}, {});
  test.Run();
}

TEST(TensorOpTest, Tile1DWithZeroRepeatsInt32Input) {
  OpTester test("Tile");

  test.AddInput<int32_t>("input", {3}, {1, 2, 3});
  test.AddInput<int64_t>("repeats", {1}, {0});
  test.AddOutput<int32_t>("output", {0}, {});
  test.Run();
}

TEST(TensorOpTest, Tile2DWithZeroRepeatsFloatInput) {
  OpTester test("Tile");

  test.AddInput<float>("input", {2, 2},
                       {11.0f, 12.0f,
                        21.0f, 22.0f});
  test.AddInput<int64_t>("repeats", {2}, {2, 0});
  test.AddOutput<float>("output", {4, 0}, {});
  test.Run();
}

TEST(TensorOpTest, Tile2DWithZeroRepeatsInt32Input) {
  OpTester test("Tile");

  test.AddInput<int32_t>("input", {2, 2},
                       {11, 12,
                        21, 22});
  test.AddInput<int64_t>("repeats", {2}, {2, 0});
  test.AddOutput<int32_t>("output", {4, 0}, {});
  test.Run();
}

TEST(TensorOpTest, Tile1DFloatInput) {
  OpTester test("Tile");

  test.AddInput<float>("input", {3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("repeats", {1}, {3});
  test.AddOutput<float>("output", {9}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(TensorOpTest, Tile1DInt32Input) {
  OpTester test("Tile");

  test.AddInput<int32_t>("input", {3}, {1, 2, 3});
  test.AddInput<int64_t>("repeats", {1}, {3});
  test.AddOutput<int32_t>("output", {9}, {1, 2, 3, 1, 2, 3, 1, 2, 3});
  test.Run();
}

TEST(TensorOpTest, Tile2D_1AxisFloatInput) {
  OpTester test("Tile");

  test.AddInput<float>("input", {2, 2},
                       {11.0f, 12.0f,
                        21.0f, 22.0f});
  test.AddInput<int64_t>("repeats", {2}, {2, 1});
  test.AddOutput<float>("output", {4, 2},
                        {11.0f, 12.0f,
                         21.0f, 22.0f,
                         11.0f, 12.0f,
                         21.0f, 22.0f});

  test.Run();
}

TEST(TensorOpTest, Tile2D_1AxisInt32Input) {
  OpTester test("Tile");

  test.AddInput<int32_t>("input", {2, 2},
                       {11, 12,
                        21, 22});
  test.AddInput<int64_t>("repeats", {2}, {2, 1});
  test.AddOutput<int32_t>("output", {4, 2},
                        {11, 12,
                         21, 22,
                         11, 12,
                         21, 22});

  test.Run();
}

TEST(TensorOpTest, Tile2D_2AxesFloatInput) {
  OpTester test("Tile");

  test.AddInput<float>("input", {2, 2},
                       {11.0f, 12.0f,
                        21.0f, 22.0f});
  test.AddInput<int64_t>("repeats", {2}, {2, 2});
  test.AddOutput<float>("output", {4, 4},
                        {11.0f, 12.0f, 11.0f, 12.0f,
                         21.0f, 22.0f, 21.0f, 22.0f,
                         11.0f, 12.0f, 11.0f, 12.0f,
                         21.0f, 22.0f, 21.0f, 22.0f});

  test.Run();
}

TEST(TensorOpTest, Tile2D_2AxesInt32Input) {
  OpTester test("Tile");

  test.AddInput<int32_t>("input", {2, 2},
                       {11, 12,
                        21, 22});
  test.AddInput<int64_t>("repeats", {2}, {2, 2});
  test.AddOutput<int32_t>("output", {4, 4},
                        {11, 12, 11, 12,
                         21, 22, 21, 22,
                         11, 12, 11, 12,
                         21, 22, 21, 22});

  test.Run();
}

TEST(TensorOpTest, Tile3DFloatInput) {
  OpTester test("Tile");

  test.AddInput<float>("input", {2, 1, 3},
                       {111.0f, 112.0f, 113.0f,
                        211.0f, 212.0f, 213.0f});
  test.AddInput<int64_t>("repeats", {3}, {1, 2, 1});
  test.AddOutput<float>("output", {2, 2, 3},
                        {111.0f, 112.0f, 113.0f,
                         111.0f, 112.0f, 113.0f,

                         211.0f, 212.0f, 213.0f,
                         211.0f, 212.0f, 213.0f});
  test.Run();
}

TEST(TensorOpTest, Tile3DInt32Input) {
  OpTester test("Tile");

  test.AddInput<int32_t>("input", {2, 1, 3},
                       {111, 112, 113,
                        211, 212, 213});
  test.AddInput<int64_t>("repeats", {3}, {1, 2, 1});
  test.AddOutput<int32_t>("output", {2, 2, 3},
                        {111, 112, 113,
                         111, 112, 113,

                         211, 212, 213,
                         211, 212, 213});
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
