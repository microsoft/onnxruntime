// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/roi_pool.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(RoIPoolTest, MaxRoiPool) {
  OpTester test("MaxRoiPool");

  constexpr int64_t pooled_height = 1, pooled_width = 1;
  test.AddAttribute("pooled_shape", std::vector<int64_t>{pooled_height, pooled_width});

  constexpr int H = 6, W = 6;
  constexpr int image_size = H * W;
  constexpr int input_channels = 3;
  std::vector<float> input;
  for (int i = 0; i < input_channels * image_size; i++)
    input.push_back(1.0f * i / 10);
  std::vector<float> rois = {
      0, 1, 1, 2, 3,
      0, 1, 1, 2, 3,
      0, 1, 1, 2, 3};

  std::vector<int64_t> x_dims = {1, 3, H, W};

  test.AddInput<float>("X", x_dims, input);
  std::vector<int64_t> rois_dims = {3, 5};
  test.AddInput<float>("rois", rois_dims, rois);

  const std::vector<float> expected_vals = {
      2.0f, 5.6f, 9.2f,
      2.0f, 5.6f, 9.2f,
      2.0f, 5.6f, 9.2f};
  std::vector<int64_t> expected_dims = {3, 3, pooled_height, pooled_width};
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
