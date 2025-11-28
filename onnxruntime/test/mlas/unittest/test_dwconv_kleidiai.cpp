//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#if defined(USE_KLEIDIAI) && !defined(_MSC_VER)

#include "test_util.h"
#include "core/mlas/lib/kleidiai/mlasi_kleidiai.h"

#include <algorithm>
#include <limits>
#include <random>
#include <vector>
void DepthwiseReferenceNchw(const float* input,
                            const float* weights,
                            const float* bias,
                            size_t batches,
                            size_t channels,
                            size_t in_height,
                            size_t in_width,
                            size_t filter_height,
                            size_t filter_width,
                            size_t pad_top,
                            size_t pad_left,
                            size_t pad_bottom,
                            size_t pad_right,
                            float clamp_min,
                            float clamp_max,
                            float* output) {
  const size_t out_height = in_height + pad_top + pad_bottom + 1 - filter_height;
  const size_t out_width = in_width + pad_left + pad_right + 1 - filter_width;

  for (size_t b = 0; b < batches; ++b) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t oh = 0; oh < out_height; ++oh) {
        for (size_t ow = 0; ow < out_width; ++ow) {
          float acc = bias != nullptr ? bias[c] : 0.0f;
          for (size_t kh = 0; kh < filter_height; ++kh) {
            const int in_y = static_cast<int>(oh) + static_cast<int>(kh) - static_cast<int>(pad_top);
            if (in_y < 0 || in_y >= static_cast<int>(in_height)) {
              continue;
            }
            for (size_t kw = 0; kw < filter_width; ++kw) {
              const int in_x = static_cast<int>(ow) + static_cast<int>(kw) - static_cast<int>(pad_left);
              if (in_x < 0 || in_x >= static_cast<int>(in_width)) {
                continue;
              }
              const size_t input_idx =
                  (((b * channels) + c) * in_height + static_cast<size_t>(in_y)) * in_width + static_cast<size_t>(in_x);
              const size_t weight_idx = (c * filter_height + kh) * filter_width + kw;
              acc += input[input_idx] * weights[weight_idx];
            }
          }
          const size_t output_idx = (((b * channels) + c) * out_height + oh) * out_width + ow;
          output[output_idx] = std::clamp(acc, clamp_min, clamp_max);
        }
      }
    }
  }
}

void RunDepthwiseConvCase(size_t channels, size_t in_height, size_t in_width, size_t padding) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
    GTEST_SKIP() << "DepthwiseConvKleidiAI requires ARM64 SME2. Skipping test.";
  }

  constexpr size_t batches = 1;
  constexpr size_t filter_height = 3;
  constexpr size_t filter_width = 3;

  const size_t pad_top = padding;
  const size_t pad_left = padding;
  const size_t pad_bottom = padding;
  const size_t pad_right = padding;

  const size_t input_size = batches * channels * in_height * in_width;
  const size_t weights_size = channels * filter_height * filter_width;
  const size_t out_height = in_height + pad_top + pad_bottom + 1 - filter_height;
  const size_t out_width = in_width + pad_left + pad_right + 1 - filter_width;
  const size_t output_size = batches * channels * out_height * out_width;

  std::vector<float> input(input_size);
  std::vector<float> weights(weights_size);
  std::vector<float> bias(channels);
  std::vector<float> expected(output_size);
  std::vector<float> output(output_size, std::numeric_limits<float>::quiet_NaN());

  std::mt19937 rng(static_cast<uint32_t>(channels * 131 + in_height * 17 + padding));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  auto fill_buffer = [&](std::vector<float>& buffer) {
    for (float& v : buffer) {
      v = dist(rng);
    }
  };

  fill_buffer(input);
  fill_buffer(weights);
  fill_buffer(bias);

  const float clamp_min = -std::numeric_limits<float>::max();
  const float clamp_max = std::numeric_limits<float>::max();

  DepthwiseReferenceNchw(input.data(),
                         weights.data(),
                         bias.data(),
                         batches,
                         channels,
                         in_height,
                         in_width,
                         filter_height,
                         filter_width,
                         pad_top,
                         pad_left,
                         pad_bottom,
                         pad_right,
                         clamp_min,
                         clamp_max,
                         expected.data());

  const bool status = DepthwiseConvKleidiAI(batches,
                                            in_height,
                                            in_width,
                                            channels,
                                            filter_height,
                                            filter_width,
                                            pad_top,
                                            pad_left,
                                            pad_bottom,
                                            pad_right,
                                            input.data(),
                                            weights.data(),
                                            bias.data(),
                                            output.data(),
                                            clamp_min,
                                            clamp_max);
  ASSERT_TRUE(status);

  for (size_t i = 0; i < output_size; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-4f) << "Mismatch at element " << i;
  }
}

TEST(MlasKleidiDepthwiseTest, ZeroPadding) {
  RunDepthwiseConvCase(/*channels=*/32, /*in_height=*/8, /*in_width=*/8, /*padding=*/0);
}

TEST(MlasKleidiDepthwiseTest, UnitPadding) {
  RunDepthwiseConvCase(/*channels=*/32, /*in_height=*/8, /*in_width=*/8, /*padding=*/1);
}

#endif  // defined(USE_KLEIDIAI) && !defined(_MSC_VER)
