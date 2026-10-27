// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "default_providers.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename T>
void RunConv2dMMInputChannelsTest(int64_t input_channels, bool weights_are_initializers) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  constexpr int64_t batch = 2;
  constexpr int64_t height = 8;
  constexpr int64_t width = 8;
  constexpr int64_t output_channels = 16;
  constexpr int64_t kernel_size = 3;
  std::vector<float> input(static_cast<size_t>(batch * input_channels * height * width));
  std::vector<float> weights(static_cast<size_t>(output_channels * input_channels * kernel_size * kernel_size));
  std::vector<float> bias(static_cast<size_t>(output_channels));
  std::vector<float> expected(static_cast<size_t>(batch * output_channels * height * width));
  // Distinct channel values expose reads from the wrong input-channel triplet.
  // Binary fractions keep both the inputs and the reference accumulation exact.
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int64_t>(i % 37) - 18) / 8.0f;
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = static_cast<float>(static_cast<int64_t>(i % 11) - 5) / 16.0f;
  }
  for (int64_t c = 0; c < output_channels; ++c) {
    bias[static_cast<size_t>(c)] = static_cast<float>(c - 8) / 8.0f;
  }
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t oc = 0; oc < output_channels; ++oc) {
      for (int64_t h = 0; h < height; ++h) {
        for (int64_t w = 0; w < width; ++w) {
          float sum = bias[static_cast<size_t>(oc)];
          for (int64_t ic = 0; ic < input_channels; ++ic) {
            for (int64_t kh = 0; kh < kernel_size; ++kh) {
              for (int64_t kw = 0; kw < kernel_size; ++kw) {
                const int64_t ih = h + kh - 1;
                const int64_t iw = w + kw - 1;
                if (ih < 0 || ih >= height || iw < 0 || iw >= width) {
                  continue;
                }
                const size_t input_index = static_cast<size_t>(((n * input_channels + ic) * height + ih) * width + iw);
                const size_t weight_index =
                    static_cast<size_t>(((oc * input_channels + ic) * kernel_size + kh) * kernel_size + kw);
                sum += input[input_index] * weights[weight_index];
              }
            }
          }
          expected[static_cast<size_t>(((n * output_channels + oc) * height + h) * width + w)] = sum;
        }
      }
    }
  }

  OpTester test("Conv", 11);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{kernel_size, kernel_size});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
  test.AddInput<T>("X", {batch, input_channels, height, width}, std::vector<T>(input.begin(), input.end()));
  test.AddInput<T>("W", {output_channels, input_channels, kernel_size, kernel_size},
                   std::vector<T>(weights.begin(), weights.end()), weights_are_initializers);
  test.AddInput<T>("B", {output_channels}, std::vector<T>(bias.begin(), bias.end()), weights_are_initializers);
  test.AddOutput<T>("Y", {batch, output_channels, height, width}, std::vector<T>(expected.begin(), expected.end()));
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

class Conv2dMMInputChannelsTest : public ::testing::TestWithParam<std::tuple<int64_t, bool>> {};

TEST_P(Conv2dMMInputChannelsTest, Float) {
  const auto [input_channels, weights_are_initializers] = GetParam();
  RunConv2dMMInputChannelsTest<float>(input_channels, weights_are_initializers);
}

TEST_P(Conv2dMMInputChannelsTest, Float16) {
  const auto [input_channels, weights_are_initializers] = GetParam();
  RunConv2dMMInputChannelsTest<MLFloat16>(input_channels, weights_are_initializers);
}

// Three channels only reads the first triplet. Larger multiples of three must
// also read subsequent triplets. Include scalar and vec4 dispatch controls.
INSTANTIATE_TEST_SUITE_P(
    Conv_WebGPU, Conv2dMMInputChannelsTest,
    ::testing::Combine(::testing::Values(int64_t{3}, int64_t{6}, int64_t{9}, int64_t{15},
                                         int64_t{5}, int64_t{4}, int64_t{12}),
                       ::testing::Bool()));

}  // namespace
}  // namespace test
}  // namespace onnxruntime
