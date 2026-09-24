// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

using std::pair;
using std::vector;

#if defined(USE_WEBGPU)

namespace {

enum class WebGpuPointwiseConvCase {
  NoBias,
  Bias,
  FusedClip,
  FusedQuickGelu,
  FusedHardSwish,
  FusedElu,
  FusedGelu,
  FusedGeluTanh,
  FusedSoftplus,
  FusedThresholdedRelu,
  FusedErf,
};

void RunWebGpuSubgroupPointwiseConvTest(WebGpuPointwiseConvCase test_case, int64_t batch = 1) {
  const bool is_fused = test_case != WebGpuPointwiseConvCase::NoBias &&
                        test_case != WebGpuPointwiseConvCase::Bias;
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  constexpr int64_t kInputChannels = 16;
  constexpr int64_t kOutputChannels = 97;
  constexpr int64_t kHeight = 4;
  constexpr int64_t kWidth = 8;
  const bool has_bias = test_case != WebGpuPointwiseConvCase::NoBias;

  // Layout transformation represents fused NHWC Conv as internal-domain Conv
  // while preserving the NhwcFusedConv activation attributes.
  OpTester test("Conv", 11,
                is_fused ? onnxruntime::kMSInternalNHWCDomain : onnxruntime::kOnnxDomain);
  test.AddAttribute("group", static_cast<int64_t>(1));
  test.AddAttribute("kernel_shape", vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("strides", vector<int64_t>{1, 1});
  test.AddAttribute("dilations", vector<int64_t>{1, 1});
  if (is_fused) {
    std::string activation;
    vector<float> activation_params;
    switch (test_case) {
      case WebGpuPointwiseConvCase::FusedClip:
        activation = "Clip";
        activation_params = {-1.0f, 2.0f};
        break;
      case WebGpuPointwiseConvCase::FusedQuickGelu:
        activation = "QuickGelu";
        activation_params = {1.4f};
        break;
      case WebGpuPointwiseConvCase::FusedHardSwish:
        activation = "HardSwish";
        break;
      case WebGpuPointwiseConvCase::FusedElu:
        activation = "Elu";
        activation_params = {0.5f};
        break;
      case WebGpuPointwiseConvCase::FusedGelu:
        activation = "Gelu";
        activation_params = {0.0f};
        break;
      case WebGpuPointwiseConvCase::FusedGeluTanh:
        activation = "Gelu";
        activation_params = {1.0f};
        break;
      case WebGpuPointwiseConvCase::FusedSoftplus:
        activation = "Softplus";
        break;
      case WebGpuPointwiseConvCase::FusedThresholdedRelu:
        activation = "ThresholdedRelu";
        activation_params = {0.6f};
        break;
      case WebGpuPointwiseConvCase::FusedErf:
        activation = "Erf";
        break;
      default:
        ORT_THROW("Unexpected non-fused WebGPU pointwise Conv case.");
    }
    test.AddAttribute("activation", activation);
    if (!activation_params.empty()) {
      test.AddAttribute("activation_params", activation_params);
    }
  }

  const vector<int64_t> x_shape = is_fused
                                      ? vector<int64_t>{batch, kHeight, kWidth, kInputChannels}
                                      : vector<int64_t>{batch, kInputChannels, kHeight, kWidth};
  vector<MLFloat16> x(static_cast<size_t>(batch * kInputChannels * kHeight * kWidth), MLFloat16(1.0f));

  const vector<int64_t> w_shape{kOutputChannels, kInputChannels, 1, 1};
  vector<MLFloat16> w;
  w.reserve(static_cast<size_t>(kOutputChannels * kInputChannels));
  vector<float> channel_values;
  channel_values.reserve(static_cast<size_t>(kOutputChannels));
  for (int64_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
    const float channel_value = static_cast<float>(output_channel % 7 - 3);
    channel_values.push_back(channel_value);
    for (int64_t input_channel = 0; input_channel < kInputChannels; ++input_channel) {
      w.emplace_back(channel_value / static_cast<float>(kInputChannels));
    }
  }

  vector<MLFloat16> bias;
  if (has_bias) {
    bias.reserve(static_cast<size_t>(kOutputChannels));
    for (int64_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
      const float bias_value = is_fused ? 0.5f : static_cast<float>(output_channel % 3 - 1) * 0.25f;
      bias.emplace_back(bias_value);
      channel_values[static_cast<size_t>(output_channel)] += bias_value;
    }
  }
  if (is_fused) {
    std::transform(channel_values.begin(), channel_values.end(), channel_values.begin(),
                   [test_case](float value) {
                     switch (test_case) {
                       case WebGpuPointwiseConvCase::FusedClip:
                         return std::clamp(value, -1.0f, 2.0f);
                       case WebGpuPointwiseConvCase::FusedQuickGelu:
                         return value / (1.0f + std::exp(-1.4f * value));
                       case WebGpuPointwiseConvCase::FusedHardSwish:
                         return value * std::clamp(value / 6.0f + 0.5f, 0.0f, 1.0f);
                       case WebGpuPointwiseConvCase::FusedElu:
                         return value >= 0.0f ? value : 0.5f * (std::exp(value) - 1.0f);
                       case WebGpuPointwiseConvCase::FusedGelu:
                         return 0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
                       case WebGpuPointwiseConvCase::FusedGeluTanh:
                         return 0.5f * value *
                                (1.0f + std::tanh(0.7978845608f *
                                                  (value + 0.044715f * value * value * value)));
                       case WebGpuPointwiseConvCase::FusedSoftplus:
                         return std::log1p(std::exp(value));
                       case WebGpuPointwiseConvCase::FusedThresholdedRelu:
                         return value > 0.6f ? value : 0.0f;
                       case WebGpuPointwiseConvCase::FusedErf:
                         return std::erf(value);
                       default:
                         ORT_THROW("Unexpected non-fused WebGPU pointwise Conv case.");
                     }
                   });
  }

  const vector<int64_t> y_shape = is_fused
                                      ? vector<int64_t>{batch, kHeight, kWidth, kOutputChannels}
                                      : vector<int64_t>{batch, kOutputChannels, kHeight, kWidth};
  vector<MLFloat16> expected;
  expected.reserve(static_cast<size_t>(batch * kOutputChannels * kHeight * kWidth));
  if (is_fused) {
    for (int64_t spatial = 0; spatial < batch * kHeight * kWidth; ++spatial) {
      for (float value : channel_values) {
        expected.emplace_back(value);
      }
    }
  } else {
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
      for (float value : channel_values) {
        expected.insert(expected.end(), static_cast<size_t>(kHeight * kWidth), MLFloat16(value));
      }
    }
  }

  test.AddInput<MLFloat16>("X", x_shape, x);
  test.AddInput<MLFloat16>("W", w_shape, w, /*is_initializer=*/true);
  if (has_bias) {
    test.AddInput<MLFloat16>("B", {kOutputChannels}, bias, /*is_initializer=*/true);
  }
  test.AddOutput<MLFloat16>("Y", y_shape, expected, /*no sort*/ false, 0.01f, 0.01f);
  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace

TEST(ConvFp16Test, WebGpuSubgroupPointwiseNoBias) {
  RunWebGpuSubgroupPointwiseConvTest(WebGpuPointwiseConvCase::NoBias);
}

TEST(ConvFp16Test, WebGpuSubgroupPointwiseBias) {
  RunWebGpuSubgroupPointwiseConvTest(WebGpuPointwiseConvCase::Bias);
}

TEST(ConvFp16Test, WebGpuSubgroupPointwiseBatch) {
  RunWebGpuSubgroupPointwiseConvTest(WebGpuPointwiseConvCase::NoBias, 2);
}

TEST(ConvFp16Test, WebGpuNaivePointwiseNhwcBias) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  OpTester test("Conv", 11, onnxruntime::kMSInternalNHWCDomain);
  test.AddAttribute("group", static_cast<int64_t>(1));
  test.AddAttribute("kernel_shape", vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("strides", vector<int64_t>{1, 1});
  test.AddAttribute("dilations", vector<int64_t>{1, 1});

  test.AddInput<MLFloat16>("X", {1, 1, 2, 4},
                           vector<MLFloat16>(8, MLFloat16(1.0f)));
  test.AddInput<MLFloat16>("W", {6, 4, 1, 1},
                           vector<MLFloat16>(24, MLFloat16(1.0f)),
                           /*is_initializer=*/true);
  test.AddInput<MLFloat16>("B", {6},
                           {MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(2.0f),
                            MLFloat16(3.0f), MLFloat16(4.0f), MLFloat16(5.0f)},
                           /*is_initializer=*/true);
  test.AddOutput<MLFloat16>("Y", {1, 1, 2, 6},
                            {MLFloat16(4.0f), MLFloat16(5.0f), MLFloat16(6.0f),
                             MLFloat16(7.0f), MLFloat16(8.0f), MLFloat16(9.0f),
                             MLFloat16(4.0f), MLFloat16(5.0f), MLFloat16(6.0f),
                             MLFloat16(7.0f), MLFloat16(8.0f), MLFloat16(9.0f)},
                            /*no_sort=*/false, 0.01f, 0.01f);
  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

#ifndef DISABLE_CONTRIB_OPS
TEST(ConvFp16Test, WebGpuSubgroupPointwiseNhwcFusedConvClip) {
  RunWebGpuSubgroupPointwiseConvTest(WebGpuPointwiseConvCase::FusedClip);
}

TEST(ConvFp16Test, WebGpuSubgroupPointwiseNhwcFusedConvNewActivations) {
  const vector<pair<const char*, WebGpuPointwiseConvCase>> cases = {
      {"QuickGelu", WebGpuPointwiseConvCase::FusedQuickGelu},
      {"HardSwish", WebGpuPointwiseConvCase::FusedHardSwish},
      {"Elu", WebGpuPointwiseConvCase::FusedElu},
      {"Gelu", WebGpuPointwiseConvCase::FusedGelu},
      {"GeluTanh", WebGpuPointwiseConvCase::FusedGeluTanh},
      {"Softplus", WebGpuPointwiseConvCase::FusedSoftplus},
      {"ThresholdedRelu", WebGpuPointwiseConvCase::FusedThresholdedRelu},
      {"Erf", WebGpuPointwiseConvCase::FusedErf},
  };

  for (const auto& [name, test_case] : cases) {
    SCOPED_TRACE(name);
    RunWebGpuSubgroupPointwiseConvTest(test_case);
  }
}
#endif

#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime
