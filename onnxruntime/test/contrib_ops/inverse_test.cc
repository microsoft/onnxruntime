// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

TEST(InverseContribOpTest, two_by_two_float) {
  OpTester test("Inverse", 1, kMSDomain);
  test.AddInput<float>("X", {2, 2}, {4, 7, 2, 6});
  test.AddOutput<float>("Y", {2, 2}, {0.6f, -0.7f, -0.2f, 0.4f});
  test.Run();
}

TEST(InverseContribOpTest, two_by_two_double) {
  OpTester test("Inverse", 1, kMSDomain);
  test.AddInput<double>("X", {2, 2}, {4, 7, 2, 6});
  test.AddOutput<double>("Y", {2, 2}, {0.6f, -0.7f, -0.2f, 0.4f});
  test.Run();
}

TEST(InverseContribOpTest, two_by_two_float16) {
  OpTester test("Inverse", 1, kMSDomain);

  auto input_float = {4.f, 7.f, 2.f, 6.f};
  std::vector<MLFloat16> input;
  std::transform(
      input_float.begin(), input_float.end(), std::back_inserter(input),
      [](float v) {
        return MLFloat16(v);
      });

  auto output_float = {0.6f, -0.7f, -0.2f, 0.4f};
  std::vector<MLFloat16> output;
  std::transform(
      output_float.begin(), output_float.end(), std::back_inserter(output), [](float v) {
        return MLFloat16(v);
      });

  test.AddInput<MLFloat16>("X", {2, 2}, input);
  test.AddOutput<MLFloat16>("Y", {2, 2}, output);
  test.Run();
}

TEST(InverseContribOpTest, four_by_four_float) {
  OpTester test("Inverse", 1, kMSDomain);
  test.AddInput<float>("X", {4, 4},
                       {4.f, 0.f, 0.f, 0.f,
                        0.f, 0.f, 2.f, 0.f,
                        0.f, 1.f, 2.f, 0.f,
                        1.f, 0.f, 0.f, 1.f});
  test.AddOutput<float>("Y", {4, 4}, {0.25f, 0.f, 0.f, 0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.5f, 0.f, 0.f, -0.25f, 0.f, 0.f, 1.f});
  test.Run();
}

TEST(InverseContribOpTest, four_by_four_batches_float) {
  OpTester test("Inverse", 1, kMSDomain);

  auto one_input_matrix_4x4 = {
      4.f, 0.f, 0.f, 0.f,
      0.f, 0.f, 2.f, 0.f,
      0.f, 1.f, 2.f, 0.f,
      1.f, 0.f, 0.f, 1.f};

  // batches 3x4 i.e. 12 batches so the full shape is 3x4x4x4
  std::vector<float> input;
  for (int64_t i = 0; i < 3 * 4; ++i) {
    std::copy(one_input_matrix_4x4.begin(), one_input_matrix_4x4.end(), std::back_inserter(input));
  }

  auto one_output_matrix_4x4 = {
      0.25f, 0.f, 0.f, 0.f,
      0.f, -1.f, 1.f, 0.f,
      0.f, 0.5f, 0.f, 0.f,
      -0.25f, 0.f, 0.f, 1.f};

  std::vector<float> output;
  for (int64_t i = 0; i < 3 * 4; ++i) {
    std::copy(one_output_matrix_4x4.begin(), one_output_matrix_4x4.end(), std::back_inserter(output));
  }

  test.AddInput<float>("Input", {3, 4, 4, 4}, input);
  test.AddOutput<float>("Output", {3, 4, 4, 4}, output);
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
