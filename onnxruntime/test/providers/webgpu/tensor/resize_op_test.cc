// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename T>
std::vector<T> Convert(const std::vector<float>& values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (float v : values) result.push_back(T(v));
  return result;
}

// WebNN conformance "resample2d(upsample) float32 4D tensor options.axes=[1, 2] options.mode='linear'":
// a channels-last [1, 2, 3, 1] input resized to [1, 4, 6, 1] (scales [1, 2, 2, 1]), half_pixel.
template <typename T>
void TestLinearResizeChannelsLast(float abs_error) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  const std::vector<float> input{59.92947006225586f, 41.98918914794922f, 66.39534759521484f,
                                 90.7006607055664f, 86.95105743408203f, 79.10005187988281f};
  const std::vector<float> expected{
      59.92947006225586f, 55.444400787353516f, 46.47425842285156f,
      48.090728759765625f, 60.29380798339844f, 66.39534759521484f,
      67.62226867675781f, 64.02411651611328f, 56.82780838012695f,
      57.31512451171875f, 65.48605346679688f, 69.57152557373047f,
      83.00786590576172f, 81.18354797363281f, 77.534912109375f,
      75.76390838623047f, 75.87055206298828f, 75.92387390136719f,
      90.7006607055664f, 89.76325988769531f, 87.88845825195312f,
      84.9883041381836f, 81.06280517578125f, 79.10005187988281f};

  OpTester test("Resize", 19);
  test.AddAttribute<std::string>("mode", "linear");
  test.AddInput<T>("X", {1, 2, 3, 1}, Convert<T>(input));
  test.AddInput<float>("roi", {0}, {});
  test.AddInput<float>("", {0}, {});  // omitted scales: the output size comes from sizes
  test.AddInput<int64_t>("sizes", {4}, {1, 4, 6, 1}, true);
  test.AddOutput<T>("Y", {1, 4, 6, 1}, Convert<T>(expected));
  test.SetOutputAbsErr("Y", abs_error);
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// Bilinear resize of a channels-last [n, h, w, c] tensor with half_pixel coordinates, in double precision.
std::vector<float> ReferenceLinearResizeChannelsLast(const std::vector<float>& x, const std::vector<int64_t>& x_dims,
                                                     const std::vector<int64_t>& y_dims) {
  const int64_t in_h = x_dims[1], in_w = x_dims[2], c = x_dims[3];
  const int64_t out_h = y_dims[1], out_w = y_dims[2];
  // Maps an output coordinate to its two source taps and the weight of the second tap.
  const auto taps = [](int64_t out, int64_t in_len, int64_t out_len) {
    const double scale = static_cast<double>(out_len) / static_cast<double>(in_len);
    const double coord =
        std::clamp((static_cast<double>(out) + 0.5) / scale - 0.5, 0.0, static_cast<double>(in_len - 1));
    const auto i0 = static_cast<int64_t>(coord);
    return std::make_tuple(i0, std::min(i0 + 1, in_len - 1), coord - static_cast<double>(i0));
  };

  std::vector<float> y;
  y.reserve(static_cast<size_t>(y_dims[0] * out_h * out_w * c));
  for (int64_t b = 0; b < y_dims[0]; ++b) {
    const auto at = [&](int64_t h, int64_t w, int64_t ch) {
      return static_cast<double>(x[static_cast<size_t>(((b * in_h + h) * in_w + w) * c + ch)]);
    };
    for (int64_t oh = 0; oh < out_h; ++oh) {
      const auto [h0, h1, wh] = taps(oh, in_h, out_h);
      for (int64_t ow = 0; ow < out_w; ++ow) {
        const auto [w0, w1, ww] = taps(ow, in_w, out_w);
        for (int64_t ch = 0; ch < c; ++ch) {
          const double top = (1.0 - ww) * at(h0, w0, ch) + ww * at(h0, w1, ch);
          const double bottom = (1.0 - ww) * at(h1, w0, ch) + ww * at(h1, w1, ch);
          y.push_back(static_cast<float>((1.0 - wh) * top + wh * bottom));
        }
      }
    }
  }
  return y;
}

// Resizes dims 1 and 2 of a channels-last input and checks it against the double-precision reference.
template <typename T>
void TestLinearResizeChannelsLastVsReference(const std::vector<int64_t>& x_dims, const std::vector<int64_t>& y_dims) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  std::vector<float> x(static_cast<size_t>(x_dims[0] * x_dims[1] * x_dims[2] * x_dims[3]));
  for (size_t i = 0; i < x.size(); ++i) {
    // Multiples of 0.25 in [-2.75, 2.75] are exact in float16, so both types share one reference.
    x[i] = static_cast<float>(static_cast<int64_t>((i * 7) % 23) - 11) * 0.25f;
  }
  const std::vector<float> y = ReferenceLinearResizeChannelsLast(x, x_dims, y_dims);

  OpTester test("Resize", 19);
  test.AddAttribute<std::string>("mode", "linear");
  test.AddInput<T>("X", x_dims, Convert<T>(x));
  test.AddInput<float>("roi", {0}, {});
  test.AddInput<float>("", {0}, {});  // omitted scales: the output size comes from sizes
  test.AddInput<int64_t>("sizes", {4}, y_dims, true);
  test.AddOutput<T>("Y", y_dims, Convert<T>(y));
  test.SetOutputAbsErr("Y", std::is_same_v<T, MLFloat16> ? 1e-2f : 1e-4f);
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// Channel counts the kernel packs into vec4 (4, 8) and vec2 (2), and an odd count that stays scalar (3).
// Each runs an upsample with a batch of 2 and a downsample, both with non-integer scales.
template <typename T>
void TestLinearResizeChannelsLastPacking() {
  for (int64_t c : {4, 8, 2, 3}) {
    SCOPED_TRACE(c);
    TestLinearResizeChannelsLastVsReference<T>({2, 3, 5, c}, {2, 5, 8, c});
    TestLinearResizeChannelsLastVsReference<T>({1, 6, 7, c}, {1, 4, 3, c});
  }
}

}  // namespace

TEST(ResizeOpTest, WebGpu_LinearChannelsLast_Float) {
  TestLinearResizeChannelsLast<float>(1e-4f);
}

TEST(ResizeOpTest, WebGpu_LinearChannelsLast_Float16) {
  TestLinearResizeChannelsLast<MLFloat16>(0.1f);
}

TEST(ResizeOpTest, WebGpu_LinearChannelsLastMultiChannel_Float) {
  TestLinearResizeChannelsLastPacking<float>();
}

TEST(ResizeOpTest, WebGpu_LinearChannelsLastMultiChannel_Float16) {
  TestLinearResizeChannelsLastPacking<MLFloat16>();
}

}  // namespace test
}  // namespace onnxruntime
