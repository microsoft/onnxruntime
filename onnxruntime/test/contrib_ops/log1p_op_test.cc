// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>
#include <random>
#include <type_traits>

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/webgpu_context.h"
#endif

namespace onnxruntime::test {

template <typename T>
static void CheckLog1p(std::unique_ptr<IExecutionProvider> ep, bool webgpu) {
  ASSERT_NE(ep, nullptr);
  std::vector<T> input;
  for (float x : {-1.0f, -0.999f, -0.75f, -0.5f, -0.125f, -1e-3f, -1e-8f, -0.0f,
                  0.0f, 1e-8f, 1e-3f, 0.125f, 0.5f, 1.0f, 10.0f, 100.0f, 10000.0f}) {
    if (webgpu && x == -1.0f) continue;  // WGSL log(0) has no portable nonfinite guarantee.
    input.emplace_back(x);
  }
  std::mt19937 generator(1234);
  std::uniform_real_distribution<float> distribution(-0.99f, 100.0f);
  for (int i = 0; i < 1024; ++i) input.emplace_back(distribution(generator));
  for (int i = -30; i <= 14; ++i) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      if (webgpu && i < -14) continue;  // WGSL permits flushing f16 subnormals.
    }
    input.emplace_back(std::ldexp(1.0f, i));
    if (i < 0) input.emplace_back(-std::ldexp(1.0f, i));
  }
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    input.push_back(std::numeric_limits<T>::max());
    input.push_back(std::nextafter(T(-1), T(0)));
  }
  if constexpr (std::is_same_v<T, float>) {
    const float minimum_normal = std::numeric_limits<float>::min();
    for (float x : {minimum_normal, std::nextafter(minimum_normal, 1.0f), 2 * minimum_normal,
                    1e-37f, std::ldexp(1.0f, -25)}) {
      input.push_back(x);
      input.push_back(-x);
    }
  }

  std::vector<T> expected;
  for (T x : input) {
    const double value = static_cast<double>(x);
    if constexpr (std::is_same_v<T, double>)
      expected.emplace_back(std::log1p(value));
    else
      expected.emplace_back(static_cast<float>(std::log1p(value)));
  }
  OpTester tester("Log1p", 1, kMSDomain);
  const std::vector<int64_t> shape{static_cast<int64_t>(input.size())};
  tester.AddInput<T>("X", shape, input);
  tester.AddOutput<T>("Y", shape, expected);
  tester.SetOutputAbsErr("Y", 0.0f);  // Tiny results must not pass as zero.
  tester.SetOutputRelErr("Y", std::is_same_v<T, double> ? 2e-15f : std::is_same_v<T, MLFloat16> ? 2e-3f
                                                               : std::is_same_v<T, BFloat16>    ? 1e-2f
                                                               : webgpu                         ? 3e-6f
                                                                                                : 2e-7f);
  SessionOptions options;
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(ep));
  tester.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  const auto fetches = tester.GetFetches();
  ASSERT_EQ(fetches.size(), 1u);
  const T* result = fetches[0].Get<Tensor>().Data<T>();
  for (size_t i = 0; i < input.size(); ++i) {
    const double x = static_cast<double>(input[i]);
    if (!webgpu && x == 0.0) {
      EXPECT_EQ(std::signbit(static_cast<double>(result[i])), std::signbit(x));
    }
    if constexpr (std::is_same_v<T, float>) {
      if (x != 0.0 && std::abs(x) <= std::ldexp(1.0, -25)) {
        EXPECT_NE(result[i], 0.0f);
        EXPECT_EQ(result[i], input[i]);
      }
    }
  }
}

#ifdef USE_CUDA
TEST(Log1pOpTest, CudaFloat) { CheckLog1p<float>(DefaultCudaExecutionProvider(), false); }
TEST(Log1pOpTest, CudaDouble) { CheckLog1p<double>(DefaultCudaExecutionProvider(), false); }
TEST(Log1pOpTest, CudaFloat16) { CheckLog1p<MLFloat16>(DefaultCudaExecutionProvider(), false); }
TEST(Log1pOpTest, CudaBFloat16) { CheckLog1p<BFloat16>(DefaultCudaExecutionProvider(), false); }
#endif
#ifdef USE_WEBGPU
TEST(Log1pOpTest, WebGpuFloat) { CheckLog1p<float>(DefaultWebGpuExecutionProvider(), true); }
TEST(Log1pOpTest, WebGpuFloat16) {
#if !defined(ORT_USE_EP_API_ADAPTERS)
  auto ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(ep, nullptr);
  const auto& context = webgpu::WebGpuContextFactory::GetContext(ep->GetDeviceId());
  if (!context.DeviceHasFeature(wgpu::FeatureName::ShaderF16)) {
    GTEST_SKIP() << "Adapter lacks shader-f16";
  }
  CheckLog1p<MLFloat16>(std::move(ep), true);
#else
  GTEST_SKIP() << "Plugin EP interface does not expose shader-f16 support";
#endif
}
#endif

}  // namespace onnxruntime::test
