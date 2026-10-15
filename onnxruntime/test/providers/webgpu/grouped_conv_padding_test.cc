// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "default_providers.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(Conv_WebGPU, GroupedConvWithPaddingUsesSignedCoordinates) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  // Two independent audio-style channels, represented as a 2D convolution with H=1.
  OpTester test("Conv", 11);
  test.AddAttribute("group", static_cast<int64_t>(2));
  test.AddAttribute("kernel_shape", std::vector<int64_t>{1, 3});
  test.AddAttribute("pads", std::vector<int64_t>{0, 1, 0, 1});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 2, 1, 3},
                       {1.0f, 2.0f, 3.0f,
                        10.0f, 20.0f, 30.0f});
  test.AddInput<float>("W", {2, 1, 1, 3},
                       {1.0f, 2.0f, 1.0f,
                        1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("Y", {1, 2, 1, 3},
                        {4.0f, 8.0f, 8.0f,
                         30.0f, 60.0f, 50.0f});

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

class GroupedConvVectorizationTest
    : public ::testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t, bool, bool>> {};

TEST_P(GroupedConvVectorizationTest, OutputVectorsStayWithinGroups) {
  const auto [groups, input_channels_per_group, output_channels_per_group, has_bias, is_nhwc] = GetParam();
  auto webgpu_ep = DefaultWebGpuExecutionProvider(is_nhwc);
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  constexpr int64_t batch = 2;
  constexpr int64_t height = 4;
  constexpr int64_t width = 5;
  constexpr int64_t kernel_height = 2;
  constexpr int64_t kernel_width = 3;
  constexpr int64_t output_height = height - kernel_height + 1;
  constexpr int64_t output_width = width - kernel_width + 1;
  const int64_t input_channels = groups * input_channels_per_group;
  const int64_t output_channels = groups * output_channels_per_group;
  std::vector<float> input(static_cast<size_t>(batch * input_channels * height * width));
  std::vector<float> weights(static_cast<size_t>(output_channels * input_channels_per_group *
                                                 kernel_height * kernel_width));
  std::vector<float> bias(static_cast<size_t>(output_channels));
  std::vector<float> expected(static_cast<size_t>(batch * output_channels * output_height * output_width));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int64_t>(i % 37) - 18) * 0.25f;
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = static_cast<float>(static_cast<int64_t>(i % 11) - 5) * 0.25f;
  }
  for (int64_t c = 0; c < output_channels; ++c) {
    bias[static_cast<size_t>(c)] = static_cast<float>(c - 3) * 0.25f;
  }
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t c = 0; c < output_channels; ++c) {
      const int64_t first_input_channel = (c / output_channels_per_group) * input_channels_per_group;
      for (int64_t h = 0; h < output_height; ++h) {
        for (int64_t w = 0; w < output_width; ++w) {
          float sum = 0.0f;
          for (int64_t ic = 0; ic < input_channels_per_group; ++ic) {
            for (int64_t kh = 0; kh < kernel_height; ++kh) {
              for (int64_t kw = 0; kw < kernel_width; ++kw) {
                const size_t input_index =
                    static_cast<size_t>(((n * input_channels + first_input_channel + ic) * height + h + kh) * width + w + kw);
                const size_t weight_index =
                    static_cast<size_t>(((c * input_channels_per_group + ic) * kernel_height + kh) * kernel_width + kw);
                sum += input[input_index] * weights[weight_index];
              }
            }
          }
          const size_t output_index = static_cast<size_t>(((n * output_channels + c) * output_height + h) * output_width + w);
          expected[output_index] = sum + (has_bias ? bias[static_cast<size_t>(c)] : 0.0f);
        }
      }
    }
  }

  OpTester test("Conv", 11);
  test.AddAttribute("group", groups);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{kernel_height, kernel_width});
  test.AddInput<float>("X", {batch, input_channels, height, width}, input);
  test.AddInput<float>("W", {output_channels, input_channels_per_group, kernel_height, kernel_width},
                       weights, true);
  if (has_bias) {
    test.AddInput<float>("B", {output_channels}, bias, true);
  }
  test.AddOutput<float>("Y", {batch, output_channels, output_height, output_width}, expected);
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

INSTANTIATE_TEST_SUITE_P(
    Conv_WebGPU, GroupedConvVectorizationTest,
    ::testing::Combine(::testing::Values(int64_t{2}, int64_t{4}),
                       ::testing::Values(int64_t{1}, int64_t{3}),
                       ::testing::Values(int64_t{1}, int64_t{3}, int64_t{4}, int64_t{5},
                                         int64_t{6}, int64_t{7}, int64_t{8}, int64_t{12}),
                       ::testing::Bool(), ::testing::Bool()));

}  // namespace test
}  // namespace onnxruntime