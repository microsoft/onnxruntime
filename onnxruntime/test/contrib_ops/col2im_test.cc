// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

template <typename T>
std::vector<T> _transpose_1dvector(std::vector<T> &input, size_t C, size_t H, size_t W)
{
    size_t n = input.size();
    if (n == 0){
        throw std::runtime_error("Invalid input");
    }
    std::vector<T> trans_vec(input);

    std::cout << "input: (";
    for(size_t i = 0; i < n; ++i)
      std::cout << trans_vec[i] << ", ";
    std::cout << ")" << std::endl;

    for(size_t c = 0; c < C; ++c)
      for(size_t i = 0; i < H; ++i)
        for(size_t j = i+1; j < W; ++j)
            std::swap(trans_vec[c*(H*W) + (H*i + j)], trans_vec[c*(H*W) + (W*j + i)]);

    std::cout << "trans_vec: (";
    for(size_t i = 0; i < n; ++i)
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
  input = _transpose_1dvector(output, 1, 5, 5);
  test.AddInput<float>("input", {1, 5, 5},  input);
  test.AddInput<int64_t>("image_shape", {2},  std::vector<int64_t>{5, 5});
  test.AddInput<int64_t>("block_shape", {2},  std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {1, 1, 5, 5}, output);
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
  input = _transpose_1dvector(output, 3, 5, 5);
  test.AddInput<float>("input", {1, 15, 5},  input);
  test.AddInput<int64_t>("image_shape", {2},  std::vector<int64_t>{5, 5});
  test.AddInput<int64_t>("block_shape", {2},  std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {1, 3, 5, 5}, output);
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
  input = _transpose_1dvector(output, 1, 5, 5);
  test.AddInput<float>("input", {1, 5, 5},  input);
  test.AddInput<int64_t>("image_shape", {3},  std::vector<int64_t>{1, 5, 5});
  test.AddInput<int64_t>("block_shape", {3},  std::vector<int64_t>{1, 1, 5});

  test.AddOutput<float>("output", {1, 1, 1, 5, 5}, output);
  test.Run();
}


}  // namespace test
}  // namespace onnxruntime
