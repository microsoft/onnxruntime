// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

template <typename T>
std::vector<T> _transpose_serialized_vector(std::vector<T> &input, size_t N, size_t C, size_t H, size_t W) {
    size_t input_size = input.size();
    if (input_size == 0) {
        throw std::runtime_error("Invalid input");
    }
    std::vector<T> trans_vec(input);

    std::cout << "input: (";
    for (size_t i = 0; i < input_size; ++i)
      std::cout << trans_vec[i] << ", ";
    std::cout << ")" << std::endl;

    for (size_t n = 0; n < N; ++n)
      for (size_t c = 0; c < C; ++c)
        for (size_t h = 0; h < H; ++h)
          for (size_t w = 0; w < W; ++w)
              trans_vec[n * (C * H * W) + c * (H * W) + (h + H * w)] = input[n * (C * H * W) + c * (H * W) + (w + W * h)];

    std::cout << "trans_vec: (";
    for (size_t i = 0; i < input_size; ++i)
      std::cout << trans_vec[i] << ", ";
    std::cout << ")" << std::endl;

    return trans_vec;
}

TEST(Col2ImContribOpTest, simple4dNCHW) {
  OpTester test("Col2Im", 1, kMSDomain);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  std::vector<float> input(25);
  std::vector<float> output(25);
  std::iota(output.begin(), output.end(), 1);
  input = _transpose_serialized_vector(output, 1, 1, 5, 5);
  test.AddInput<float>("input", {1, 5, 5},  input);
  test.AddInput<int64_t>("image_shape", {2},  std::vector<int64_t>{5, 5});
  test.AddInput<int64_t>("block_shape", {2},  std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {1, 1, 5, 5}, output);
  test.Run();
}

TEST(Col2ImContribOpTest, with2Images3channelsNonSquare4dNCHW) {
  OpTester test("Col2Im", 1, kMSDomain);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  std::vector<float> input(120);
  std::vector<float> output(120);
  std::iota(output.begin(), output.end(), 1);
  input = _transpose_serialized_vector(output, 2, 3, 4, 5);
  test.AddInput<float>("input", {2, 15, 4},  input);
  test.AddInput<int64_t>("image_shape", {2},  std::vector<int64_t>{4, 5});
  test.AddInput<int64_t>("block_shape", {2},  std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {2, 3, 4, 5}, output);
  test.Run();
}

TEST(Col2ImContribOpTest, with3channels4dNCHW) {
  OpTester test("Col2Im", 1, kMSDomain);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  std::vector<float> input(75);
  std::vector<float> output(75);
  std::iota(output.begin(), output.end(), 1);
  input = _transpose_serialized_vector(output, 1, 3, 5, 5);
  test.AddInput<float>("input", {1, 15, 5},  input);
  test.AddInput<int64_t>("image_shape", {2},  std::vector<int64_t>{5, 5});
  test.AddInput<int64_t>("block_shape", {2},  std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {1, 3, 5, 5}, output);
  test.Run();
}

TEST(Col2ImContribOpTest, with2Images3channels4dNCHW) {
  OpTester test("Col2Im", 1, kMSDomain);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  std::vector<float> input(150);
  std::vector<float> output(150);
  std::iota(output.begin(), output.end(), 1);
  input = _transpose_serialized_vector(output, 2, 3, 5, 5);
  test.AddInput<float>("input", {2, 15, 5},  input);
  test.AddInput<int64_t>("image_shape", {2},  std::vector<int64_t>{5, 5});
  test.AddInput<int64_t>("block_shape", {2},  std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {2, 3, 5, 5}, output);
  test.Run();
}

TEST(Col2ImContribOpTest, simple5dNCHWD) {
  OpTester test("Col2Im", 1, kMSDomain);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0, 0, 0});

  std::vector<float> input(25);
  std::vector<float> output(25);
  std::iota(output.begin(), output.end(), 1);
  input = _transpose_serialized_vector(output, 1, 1, 5, 5);
  test.AddInput<float>("input", {1, 5, 5},  input);
  test.AddInput<int64_t>("image_shape", {3},  std::vector<int64_t>{1, 5, 5});
  test.AddInput<int64_t>("block_shape", {3},  std::vector<int64_t>{1, 1, 5});
  test.AddOutput<float>("output", {1, 1, 1, 5, 5}, output);
  test.Run();
}

TEST(Im2ColContribOpTest, simple) {
  std::vector<float> input(24);
  std::vector<float> expected_output(24);
  std::iota(input.begin(), input.end(), 1);
  expected_output = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 13, 17, 21, 14, 18, 22, 15, 19, 23, 16, 20, 24};
  float* actual_output = new float(24);
  std::cout << "\nExpected output --> "; for (auto i=0; i < 24; ++i) std::cout <<  expected_output[i] << ", ";
    std::cout << ")" << std::endl;
  math::Im2col<float, StorageOrder::NCHW>()(
    input.data(),
    int64_t(2),
    int64_t(3),
    int64_t(4),
    int64_t(1),
    int64_t(4),
    int64_t(1),
    int64_t(1),
    int64_t(0),
    int64_t(0),
    int64_t(0),
    int64_t(0),
    int64_t(1),
    int64_t(1),
    actual_output,
    0.);

    std::cout << "\nActual output   --> "; for (auto i=0; i < 24; ++i) std::cout <<  actual_output[i] <<
      ", "; std::cout << ")" << std::endl;
    delete[] actual_output;
}

}  // namespace test
}  // namespace onnxruntime
