// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "core/util/math.h"

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
std::vector<T> TransposeSerializedVector(std::vector<T>& input, size_t N, size_t C, size_t H, size_t W) {
  size_t input_size = input.size();
  if (input_size == 0) {
    throw std::runtime_error("Invalid input");
  }
  std::vector<T> trans_vec(input);

  for (size_t n = 0; n < N; ++n)
    for (size_t c = 0; c < C; ++c)
      for (size_t h = 0; h < H; ++h)
        for (size_t w = 0; w < W; ++w)
          trans_vec[n * (C * H * W) + c * (H * W) + (h + H * w)] =
              input[n * (C * H * W) + c * (H * W) + (w + W * h)];

  return trans_vec;
}

}  // namespace

TEST(Col2ImOpTest, Simple4dNCHW) {
  OpTester test("Col2Im", 18);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  std::vector<float> input(25);
  std::vector<float> output(25);
  std::iota(output.begin(), output.end(), 1.0f);

  input = TransposeSerializedVector(output, 1, 1, 5, 5);
  test.AddInput<float>("input", {1, 5, 5}, input);
  test.AddInput<int64_t>("image_shape", {2}, std::vector<int64_t>{5, 5});
  test.AddInput<int64_t>("block_shape", {2}, std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {1, 1, 5, 5}, output);
  test.Run();
}

TEST(Col2ImOpTest, With2Images3channelsNonSquare4dNCHW) {
  OpTester test("Col2Im", 18);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  std::vector<float> input(120);
  std::vector<float> output(120);
  std::iota(output.begin(), output.end(), 1.0f);
  input = TransposeSerializedVector(output, 2, 3, 4, 5);
  test.AddInput<float>("input", {2, 15, 4}, input);
  test.AddInput<int64_t>("image_shape", {2}, std::vector<int64_t>{4, 5});
  test.AddInput<int64_t>("block_shape", {2}, std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {2, 3, 4, 5}, output);
  test.Run();
}

TEST(Col2ImOpTest, With2Images2channelsNonSquareDilationPadStride4dNCHW) {
  OpTester test("Col2Im", 18);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});
  test.AddAttribute("pads", std::vector<int64_t>{2, 2, 2, 2});

  std::vector<float> input{0., 0., 0., 0., 0., 1., 3., 5., 0., 11., 13., 15., 0., 0., 0., 0.,
                           0., 0., 0., 0., 1., 3., 5., 0., 11., 13., 15., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 21., 23., 25., 0., 31., 33., 35., 0., 0., 0., 0.,
                           0., 0., 0., 0., 21., 23., 25., 0., 31., 33., 35., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 41., 43., 45., 0., 51., 53., 55., 0., 0., 0., 0.,
                           0., 0., 0., 0., 41., 43., 45., 0., 51., 53., 55., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 61., 63., 65., 0., 71., 73., 75., 0., 0., 0., 0.,
                           0., 0., 0., 0., 61., 63., 65., 0., 71., 73., 75., 0., 0., 0., 0., 0.};
  std::vector<float> output{2., 0., 6., 0., 10.,
                            0., 0., 0., 0., 0.,
                            22., 0., 26., 0., 30.,
                            0., 0., 0., 0., 0.,
                            42., 0., 46., 0., 50.,
                            0., 0., 0., 0., 0.,
                            62., 0., 66., 0., 70.,
                            0., 0., 0., 0., 0.,
                            82., 0., 86., 0., 90.,
                            0., 0., 0., 0., 0.,
                            102., 0., 106., 0., 110.,
                            0., 0., 0., 0., 0.,
                            122., 0., 126., 0., 130.,
                            0., 0., 0., 0., 0.,
                            142., 0., 146., 0., 150.,
                            0., 0., 0., 0., 0.};
  test.AddInput<float>("input", {2, 4, 16}, input);
  test.AddInput<int64_t>("image_shape", {2}, std::vector<int64_t>{4, 5});
  test.AddInput<int64_t>("block_shape", {2}, std::vector<int64_t>{1, 2});

  test.AddOutput<float>("output", {2, 2, 4, 5}, output);
  test.Run();
}

TEST(Col2ImOpTest, With3channels4dNCHW) {
  OpTester test("Col2Im", 18);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  std::vector<float> input(75);
  std::vector<float> output(75);
  std::iota(output.begin(), output.end(), 1.0f);
  input = TransposeSerializedVector(output, 1, 3, 5, 5);
  test.AddInput<float>("input", {1, 15, 5}, input);
  test.AddInput<int64_t>("image_shape", {2}, std::vector<int64_t>{5, 5});
  test.AddInput<int64_t>("block_shape", {2}, std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {1, 3, 5, 5}, output);
  test.Run();
}

TEST(Col2ImOpTest, With2Images3channels4dNCHW) {
  OpTester test("Col2Im", 18);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  std::vector<float> input(150);
  std::vector<float> output(150);
  std::iota(output.begin(), output.end(), 1.0f);
  input = TransposeSerializedVector(output, 2, 3, 5, 5);
  test.AddInput<float>("input", {2, 15, 5}, input);
  test.AddInput<int64_t>("image_shape", {2}, std::vector<int64_t>{5, 5});
  test.AddInput<int64_t>("block_shape", {2}, std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {2, 3, 5, 5}, output);
  test.Run();
}

TEST(Col2ImOpTest, Simple5dNCHWD) {
  OpTester test("Col2Im", 18);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0, 0, 0});

  std::vector<float> input(25);
  std::vector<float> output(25);
  std::iota(output.begin(), output.end(), 1.0f);
  input = TransposeSerializedVector(output, 1, 1, 5, 5);
  test.AddInput<float>("input", {1, 5, 5}, input);
  test.AddInput<int64_t>("image_shape", {3}, std::vector<int64_t>{1, 5, 5});
  test.AddInput<int64_t>("block_shape", {3}, std::vector<int64_t>{1, 1, 5});
  test.AddOutput<float>("output", {1, 1, 1, 5, 5}, output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
