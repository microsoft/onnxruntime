// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
namespace onnxruntime::test {
#ifdef USE_WEBGPU
namespace {
std::unique_ptr<IExecutionProvider> CreateWebGpuProvider(bool nchw = false) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.validationMode", "full"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.storageBufferCacheMode", "disabled"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.enableInt64", "1"));
  if (nchw) {
    ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.preferredLayout", "NCHW"));
  }
  return WebGpuExecutionProviderWithOptions(options);
}
void RunWebGpu(OpTester& test, std::unique_ptr<IExecutionProvider> provider) {
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  options.graph_optimization_level = TransformerLevel::Default;
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}
}  // namespace

TEST(WebGpuNormalizationTest, InstanceNormHalfConstant) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("InstanceNormalization", 13);
  test.AddInput<MLFloat16>("X", {1, 1, 4}, std::vector<MLFloat16>(4, MLFloat16(1000.f)));
  test.AddInput<MLFloat16>("scale", {1}, {MLFloat16(1.f)});
  test.AddInput<MLFloat16>("B", {1}, {MLFloat16(0.f)});
  test.AddOutput<MLFloat16>("Y", {1, 1, 4}, std::vector<MLFloat16>(4, MLFloat16(0.f)));
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuNormalizationTest, InstanceNormSmallEpsilon) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("InstanceNormalization", 13);
  test.AddAttribute<float>("epsilon", 1e-8f);
  test.AddInput<float>("X", {1, 1, 4}, {1, 1, 1, 1});
  test.AddInput<float>("scale", {1}, {1});
  test.AddInput<float>("B", {1}, {0});
  test.AddOutput<float>("Y", {1, 1, 4}, {0, 0, 0, 0});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuNormalizationTest, InstanceNormLargeOffset) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("InstanceNormalization", 13);
  test.AddInput<float>("X", {1, 1, 4}, {10000, 10001, 10000, 10001});
  test.AddInput<float>("scale", {1}, {1});
  test.AddInput<float>("B", {1}, {0});
  test.AddOutput<float>("Y", {1, 1, 4}, {-.99998f, .99998f, -.99998f, .99998f});
  RunWebGpu(test, std::move(ep));
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuNormalizationTest, SkipLayerNormLargeOffset) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("SkipLayerNormalization", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddInput<float>("X", {1, 1, 4}, {10000, 10001, 10000, 10001});
  test.AddInput<float>("skip", {1, 1, 4}, {0, 0, 0, 0});
  test.AddInput<float>("gamma", {4}, {1, 1, 1, 1});
  test.AddOutput<float>("Y", {1, 1, 4}, {-.99998f, .99998f, -.99998f, .99998f});
  RunWebGpu(test, std::move(ep));
}
#endif

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuNormalizationTest, SkipLayerNormCenteredSplitWithResidualOutput) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("SkipLayerNormalization", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-5f);
  std::vector<float> x(1024), expected(1024);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = 10000.f + (i % 2);
    expected[i] = (i % 2 ? 1.f : -1.f) / std::sqrt(1.00004f);
  }
  test.AddInput<float>("X", {1, 1, 1024}, x);
  test.AddInput<float>("skip", {1, 1, 1024}, std::vector<float>(1024, 0));
  test.AddInput<float>("gamma", {1024}, std::vector<float>(1024, 1));
  test.AddOutput<float>("Y", {1, 1, 1024}, expected);
  test.AddOptionalOutputEdge<float>();
  test.AddOptionalOutputEdge<float>();
  test.AddOutput<float>("sum", {1, 1, 1024}, x);
  RunWebGpu(test, std::move(ep));
}
#endif

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuNormalizationTest, GroupNormCenteredVariance) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("GroupNorm", 1, kMSDomain);
  test.AddAttribute<int64_t>("groups", 1);
  test.AddAttribute<int64_t>("channels_last", 1);
  test.AddAttribute<int64_t>("activation", 0);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddInput<float>("X", {1, 1, 1, 4}, {10000, 10001, 10000, 10001});
  test.AddInput<float>("gamma", {4}, {1, 1, 1, 1});
  test.AddInput<float>("beta", {4}, {0, 0, 0, 0});
  test.AddOutput<float>("Y", {1, 1, 1, 4}, {-.99998f, .99998f, -.99998f, .99998f});
  RunWebGpu(test, std::move(ep));
}
#endif

TEST(WebGpuNormalizationTest, CenteredNormalizationFractionalConstants) {
  for (const int64_t length : {3, 7, 33, 512, 1024}) {
    for (const bool instance : {false, true}) {
#ifdef DISABLE_CONTRIB_OPS
      if (!instance) continue;
#endif
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      SCOPED_TRACE(testing::Message() << "length=" << length << " instance=" << instance);
      OpTester test(instance ? "InstanceNormalization" : "SkipLayerNormalization", instance ? 13 : 1,
                    instance ? kOnnxDomain : kMSDomain);
      test.AddAttribute<float>("epsilon", 1e-5f);
      test.AddInput<float>("X", {1, 1, length}, std::vector<float>(static_cast<size_t>(length), 4080.11767578125f));
      if (instance) {
        test.AddInput<float>("scale", {1}, {1000});
        test.AddInput<float>("bias", {1}, {0});
      } else {
        test.AddInput<float>("skip", {1, 1, length}, std::vector<float>(static_cast<size_t>(length), 0));
        test.AddInput<float>("gamma", {length}, std::vector<float>(static_cast<size_t>(length), 1000));
      }
      test.AddOutput<float>("Y", {1, 1, length}, std::vector<float>(static_cast<size_t>(length), 0));
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuNormalizationTest, CenteredNormalizationCancellationAndScale) {
  for (const int64_t length : {4, 512}) {
    for (const bool instance : {false, true}) {
#ifdef DISABLE_CONTRIB_OPS
      if (!instance) continue;
#endif
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      SCOPED_TRACE(testing::Message() << "length=" << length << " instance=" << instance);
      std::vector<float> input(static_cast<size_t>(length), 0);
      input[0] = 1e8f;
      input[2] = -1e8f;
      input[3] = 1;
      const double mean = 1.0 / length;
      double variance = 0;
      for (float value : input) variance += (value - mean) * (value - mean);
      variance /= length;
      std::vector<float> expected;
      for (float value : input) expected.push_back(static_cast<float>((value - mean) / std::sqrt(variance + 1e-5) * 1e8));
      OpTester test(instance ? "InstanceNormalization" : "SkipLayerNormalization", instance ? 13 : 1,
                    instance ? kOnnxDomain : kMSDomain);
      test.AddAttribute<float>("epsilon", 1e-5f);
      test.AddInput<float>("X", {1, 1, length}, input);
      if (instance) {
        test.AddInput<float>("scale", {1}, {1e8f});
        test.AddInput<float>("bias", {1}, {0});
      } else {
        test.AddInput<float>("skip", {1, 1, length}, std::vector<float>(static_cast<size_t>(length), 0));
        test.AddInput<float>("gamma", {length}, std::vector<float>(static_cast<size_t>(length), 1e8f));
      }
      test.AddOutput<float>("Y", {1, 1, length}, expected);
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuNormalizationTest, CenteredNormalizationKeepsSmallMeanOffsets) {
  for (const int kind : {0, 1, 2}) {
    SCOPED_TRACE(kind);
#ifdef DISABLE_CONTRIB_OPS
    if (kind != 0) continue;
#endif
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    const float low = 4080.11767578125f;
    const float high = std::nextafter(low, std::numeric_limits<float>::infinity());
    const double half_delta = (static_cast<double>(high) - low) / 2;
    const float expected = static_cast<float>(half_delta / std::sqrt(half_delta * half_delta + 1e-5) * 1000);
    OpTester test(kind == 0 ? "InstanceNormalization" : kind == 1 ? "SkipLayerNormalization"
                                                                  : "GroupNorm",
                  kind == 0 ? 13 : 1, kind == 0 ? kOnnxDomain : kMSDomain);
    test.AddAttribute<float>("epsilon", 1e-5f);
    const std::vector<int64_t> shape = kind == 2 ? std::vector<int64_t>{1, 1, 1, 4} : std::vector<int64_t>{1, 1, 4};
    test.AddInput<float>("X", shape, {low, high, low, high});
    if (kind == 0) {
      test.AddInput<float>("scale", {1}, {1000});
      test.AddInput<float>("bias", {1}, {0});
    } else {
      if (kind == 1)
        test.AddInput<float>("skip", shape, {0, 0, 0, 0});
      else {
        test.AddAttribute<int64_t>("groups", 1);
        test.AddAttribute<int64_t>("channels_last", 1);
        test.AddAttribute<int64_t>("activation", 0);
      }
      test.AddInput<float>("gamma", {4}, {1000, 1000, 1000, 1000});
      if (kind == 2) test.AddInput<float>("beta", {4}, {0, 0, 0, 0});
    }
    test.AddOutput<float>("Y", shape, {-expected, expected, -expected, expected});
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuNormalizationTest, InstanceNormLayoutsAndVectorWidths) {
  for (const bool nchw : {false, true}) {
    for (const int64_t channels : {1, 2, 3, 4}) {
      for (const int64_t width : {3, 4}) {
        SCOPED_TRACE(testing::Message() << "nchw=" << nchw << " channels=" << channels << " width=" << width);
        auto ep = CreateWebGpuProvider(nchw);
        if (!ep) GTEST_SKIP();
        // The EP intentionally keeps standard InstanceNorm in NCHW regardless of preference.
        OpTester test("InstanceNormalization", 13, nchw ? kOnnxDomain : kMSInternalNHWCDomain);
        test.AddAttribute<float>("epsilon", 1e-5f);
        std::vector<float> input(static_cast<size_t>(2 * channels * width)), expected(input.size());
        std::vector<float> bias(static_cast<size_t>(channels));
        for (int64_t n = 0; n < 2; ++n) {
          for (int64_t c = 0; c < channels; ++c) {
            bias[static_cast<size_t>(c)] = static_cast<float>(c + 1);
            for (int64_t w = 0; w < width; ++w) {
              const auto index = static_cast<size_t>(nchw ? (n * channels + c) * width + w
                                                          : (n * width + w) * channels + c);
              input[index] = 4000.11767578125f + static_cast<float>(100 * n + 10 * c);
              expected[index] = static_cast<float>(c + 1);
            }
          }
        }
        const std::vector<int64_t> shape = nchw ? std::vector<int64_t>{2, channels, 1, width}
                                                : std::vector<int64_t>{2, 1, width, channels};
        test.AddInput<float>("X", shape, input);
        test.AddInput<float>("scale", {channels}, std::vector<float>(static_cast<size_t>(channels), 1000.f));
        test.AddInput<float>("bias", {channels}, bias);
        test.AddOutput<float>("Y", shape, expected);
        RunWebGpu(test, std::move(ep));
      }
    }
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuNormalizationTest, SimplifiedSkipNormSplitResidualControl) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("SkipSimplifiedLayerNormalization", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", 0.f);
  test.AddInput<float>("X", {1, 1, 512}, std::vector<float>(512, 1));
  test.AddInput<float>("skip", {1, 1, 512}, std::vector<float>(512, 1));
  test.AddInput<float>("gamma", {512}, std::vector<float>(512, 1));
  test.AddOutput<float>("Y", {1, 1, 512}, std::vector<float>(512, 1));
  test.AddOptionalOutputEdge<float>();
  test.AddOptionalOutputEdge<float>();
  test.AddOutput<float>("sum", {1, 1, 512}, std::vector<float>(512, 2));
  RunWebGpu(test, std::move(ep));
}
#endif

TEST(WebGpuNormalizationTest, CenteredNormalizationLargeFiniteConstants) {
  for (const int kind : {0, 1, 2}) {
#ifdef DISABLE_CONTRIB_OPS
    if (kind != 0) continue;
#endif
    SCOPED_TRACE(kind);
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test(kind == 0 ? "InstanceNormalization" : kind == 1 ? "SkipLayerNormalization"
                                                                  : "GroupNorm",
                  kind == 0 ? 13 : 1, kind == 0 ? kOnnxDomain : kMSDomain);
    test.AddAttribute<float>("epsilon", 1e-5f);
    const std::vector<int64_t> shape = kind == 2 ? std::vector<int64_t>{1, 1, 1, 4} : std::vector<int64_t>{1, 1, 4};
    test.AddInput<float>("X", shape, std::vector<float>(4, 3e30f));
    if (kind == 0) {
      test.AddInput<float>("scale", {1}, {1});
      test.AddInput<float>("bias", {1}, {0});
    } else {
      if (kind == 1)
        test.AddInput<float>("skip", shape, std::vector<float>(4, 0));
      else {
        test.AddAttribute<int64_t>("groups", 1);
        test.AddAttribute<int64_t>("channels_last", 1);
        test.AddAttribute<int64_t>("activation", 0);
      }
      test.AddInput<float>("gamma", {4}, {1, 1, 1, 1});
      if (kind == 2) test.AddInput<float>("beta", {4}, {0, 0, 0, 0});
    }
    test.AddOutput<float>("Y", shape, std::vector<float>(4, 0));
    RunWebGpu(test, std::move(ep));
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuNormalizationTest, SkipGroupNormBroadcastResidual) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("SkipGroupNorm", 1, kMSDomain);
  test.AddAttribute<int64_t>("groups", 2);
  test.AddAttribute<int64_t>("channels_last", 1);
  test.AddAttribute<int64_t>("activation", 0);
  test.AddAttribute<float>("epsilon", 1e-5f);
  const std::vector<float> input{10000, 10000, 20000, 20000, 10001, 10001, 20001, 20001,
                                 30000, 30000, 40000, 40000, 30001, 30001, 40001, 40001};
  const std::vector<float> sum{10002, 10002, 20004, 20004, 10003, 10003, 20005, 20005,
                               30004, 30004, 40006, 40006, 30005, 30005, 40007, 40007};
  const float normalized = 1.f / std::sqrt(1.00004f);
  std::vector<float> expected;
  for (int n = 0; n < 2; ++n) {
    for (int w = 0; w < 2; ++w) {
      for (int c = 0; c < 4; ++c) expected.push_back(static_cast<float>(c + 5) + (w ? normalized : -normalized) * static_cast<float>(c + 1));
    }
  }
  test.AddInput<float>("X", {2, 1, 2, 4}, input);
  test.AddInput<float>("gamma", {4}, {1, 2, 3, 4});
  test.AddInput<float>("beta", {4}, {5, 6, 7, 8});
  test.AddInput<float>("skip", {2, 4}, {1, 1, 2, 2, 3, 3, 4, 4});
  test.AddInput<float>("bias", {4}, {1, 1, 2, 2});
  test.AddOutput<float>("Y", {2, 1, 2, 4}, expected);
  test.AddOutput<float>("sum", {2, 1, 2, 4}, sum);
  RunWebGpu(test, std::move(ep));
}
#endif

#endif
}  // namespace onnxruntime::test
