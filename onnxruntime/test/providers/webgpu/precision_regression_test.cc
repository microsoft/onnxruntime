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

TEST(WebGpuPrecisionTest, BatchNormMixedTypes) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("BatchNormalization", 15);
  test.AddAttribute<float>("epsilon", 0.f);
  test.AddInput<MLFloat16>("X", {1, 1, 4}, {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f)});
  test.AddInput<float>("scale", {1}, {1});
  test.AddInput<float>("B", {1}, {0});
  test.AddInput<float>("mean", {1}, {0});
  test.AddInput<float>("var", {1}, {1});
  test.AddOutput<MLFloat16>("Y", {1, 1, 4}, {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f)});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuPrecisionTest, SoftmaxHalfLongRow) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Softmax", 13);
  test.AddInput<MLFloat16>("X", {1, 65536}, std::vector<MLFloat16>(65536, MLFloat16(0.f)));
  test.AddOutput<MLFloat16>("Y", {1, 65536}, std::vector<MLFloat16>(65536, MLFloat16(1.f / 65536)));
  test.SetOutputAbsErr("Y", 1e-7f);
  test.SetOutputRelErr("Y", 0.f);
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuPrecisionTest, AveragePoolHalfOverflow) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("AveragePool", 19);
  test.AddAttribute<std::vector<int64_t>>("kernel_shape", {2});
  test.AddInput<MLFloat16>("X", {1, 1, 2}, {MLFloat16(40000.f), MLFloat16(40000.f)});
  test.AddOutput<MLFloat16>("Y", {1, 1, 1}, {MLFloat16(40000.f)});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuPrecisionTest, DequantizeInt32ToHalfRange) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("DequantizeLinear", 19);
  test.AddInput<int32_t>("X", {1}, {131072});
  test.AddInput<MLFloat16>("scale", {}, {MLFloat16(1.f / 16384)});
  test.AddOutput<MLFloat16>("Y", {1}, {MLFloat16(8.f)});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuPrecisionTest, SoftmaxHalfLongRowNormalPeak) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Softmax", 13);
  const size_t n = 131072;
  std::vector<MLFloat16> x(n, MLFloat16(0.f));
  x.back() = MLFloat16(16.f);
  const float low = std::exp(-16.f) / (1.f + (n - 1) * std::exp(-16.f));
  std::vector<MLFloat16> y(n, MLFloat16(low));
  y.back() = MLFloat16(1.f / (1.f + (n - 1) * std::exp(-16.f)));
  test.AddInput<MLFloat16>("X", {1, static_cast<int64_t>(n)}, x);
  test.AddOutput<MLFloat16>("Y", {1, static_cast<int64_t>(n)}, y);
  test.SetOutputAbsErr("Y", 1e-3f);
  test.SetOutputRelErr("Y", 0.f);
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuPrecisionTest, AveragePoolHalfParallelRange) {
  for (const int64_t length : {128, 65536}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("AveragePool", 19);
    test.AddAttribute<std::vector<int64_t>>("kernel_shape", {length});
    test.AddInput<MLFloat16>("X", {1, 1, length}, std::vector<MLFloat16>(static_cast<size_t>(length), MLFloat16(2.f)));
    test.AddOutput<MLFloat16>("Y", {1, 1, 1}, {MLFloat16(2.f)});
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuPrecisionTest, BatchNormMixedStatisticsLayouts) {
  for (const bool nchw : {false, true}) {
    auto ep = CreateWebGpuProvider(nchw);
    if (!ep) GTEST_SKIP();
    OpTester test("BatchNormalization", 14);
    test.AddAttribute<float>("epsilon", 0.f);
    test.AddInput<MLFloat16>("X", {1, 2, 1, 2}, {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f)});
    test.AddInput<MLFloat16>("scale", {2}, {MLFloat16(2.f), MLFloat16(3.f)});
    test.AddInput<MLFloat16>("bias", {2}, {MLFloat16(1.f), MLFloat16(-1.f)});
    test.AddInput<float>("mean", {2}, {0, 1});
    test.AddInput<float>("variance", {2}, {1, 4});
    test.AddOutput<MLFloat16>("Y", {1, 2, 1, 2}, {MLFloat16(3.f), MLFloat16(5.f), MLFloat16(2.f), MLFloat16(3.5f)});
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuPrecisionTest, DequantizeInt32HalfVectorWidths) {
  for (const int64_t length : {2, 4}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("DequantizeLinear", 19);
    test.AddInput<int32_t>("X", {length}, std::vector<int32_t>(static_cast<size_t>(length), -131072));
    test.AddInput<MLFloat16>("scale", {}, {MLFloat16(1.f / 16384)});
    test.AddOutput<MLFloat16>("Y", {length}, std::vector<MLFloat16>(static_cast<size_t>(length), MLFloat16(-8.f)));
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuPrecisionTest, BatchNormHalfIntermediateRange) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("BatchNormalization", 15);
  test.AddAttribute<float>("epsilon", 0.f);
  test.AddInput<MLFloat16>("X", {1, 1, 4}, std::vector<MLFloat16>(4, MLFloat16(65504.f)));
  test.AddInput<MLFloat16>("scale", {1}, {MLFloat16(0.5f)});
  test.AddInput<MLFloat16>("bias", {1}, {MLFloat16(0.f)});
  test.AddInput<MLFloat16>("mean", {1}, {MLFloat16(-65504.f)});
  test.AddInput<MLFloat16>("variance", {1}, {MLFloat16(1.f)});
  test.AddOutput<MLFloat16>("Y", {1, 1, 4}, std::vector<MLFloat16>(4, MLFloat16(65504.f)));
  RunWebGpu(test, std::move(ep));
}

#endif
}  // namespace onnxruntime::test
