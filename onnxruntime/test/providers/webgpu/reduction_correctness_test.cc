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

TEST(WebGpuReductionTest, ReduceLogSumExpOverflow) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("ReduceLogSumExp", 13);
  test.AddInput<float>("X", {2}, {100, 100});
  test.AddOutput<float>("Y", {1}, {100.69314718f});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuReductionTest, ReduceSumInt32NaivePrecision) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("ReduceSum", 13);
  std::vector<int32_t> x(1025 * 2);
  for (size_t i = 0; i < 1025; ++i) {
    x[2 * i] = 16777217;
    x[2 * i + 1] = 1;
  }
  test.AddInput<int32_t>("X", {1025, 2}, x);
  test.AddInput<int64_t>("axes", {1}, {1});
  test.AddOutput<int32_t>("Y", {1025, 1}, std::vector<int32_t>(1025, 16777218));
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuReductionTest, ReduceLogSumExpNaiveLargeValues) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("ReduceLogSumExp", 13);
  test.AddAttribute<std::vector<int64_t>>("axes", {1});
  test.AddInput<float>("X", {1025, 2}, std::vector<float>(2050, 100));
  test.AddOutput<float>("Y", {1025, 1}, std::vector<float>(1025, 100 + std::log(2.f)));
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuReductionTest, ReduceLogSumExpInfinities) {
  for (const float v : {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("ReduceLogSumExp", 13);
    test.AddInput<float>("X", {3}, {v, v, v});
    test.AddOutput<float>("Y", {1}, {v});
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuReductionTest, ReduceIntegerNaiveSumMeanProd) {
  for (const std::string op : {"ReduceSum", "ReduceMean", "ReduceProd"}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test(op.c_str(), 13);
    if (op != "ReduceSum") test.AddAttribute<std::vector<int64_t>>("axes", {1});
    std::vector<int32_t> input(2050);
    for (size_t i = 0; i < 1025; ++i) {
      input[2 * i] = 16777217;
      input[2 * i + 1] = op == "ReduceMean" ? 16777219 : 1;
    }
    test.AddInput<int32_t>("X", {1025, 2}, input);
    if (op == "ReduceSum") test.AddInput<int64_t>("axes", {1}, {1});
    test.AddOutput<int32_t>("Y", {1025, 1}, std::vector<int32_t>(1025, op == "ReduceProd" ? 16777217 : 16777218));
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuReductionTest, IntegerMeanAvoidsOverflowInBothReductionPaths) {
  for (const int64_t rows : {1, 1025}) {
    for (const bool negative : {false, true}) {
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      OpTester test("ReduceMean", 13);
      test.AddAttribute<std::vector<int64_t>>("axes", {1});
      const int32_t value = negative ? INT32_MIN : INT32_MAX;
      test.AddInput<int32_t>("X", {rows, 2}, std::vector<int32_t>(static_cast<size_t>(rows * 2), value));
      test.AddOutput<int32_t>("Y", {rows, 1}, std::vector<int32_t>(static_cast<size_t>(rows), value));
      RunWebGpu(test, std::move(ep));
    }
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("ReduceMean", 13);
    test.AddAttribute<std::vector<int64_t>>("axes", {1});
    test.AddInput<uint32_t>("X", {rows, 2}, std::vector<uint32_t>(static_cast<size_t>(rows * 2), UINT32_MAX));
    test.AddOutput<uint32_t>("Y", {rows, 1}, std::vector<uint32_t>(static_cast<size_t>(rows), UINT32_MAX));
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuReductionTest, IntegerMeanRetainsLowBits) {
  for (const int64_t rows : {1, 1025}) {
    for (const int32_t value : {-16777217, 16777217}) {
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      OpTester test("ReduceMean", 13);
      test.AddAttribute<std::vector<int64_t>>("axes", {1});
      test.AddInput<int32_t>("X", {rows, 2}, std::vector<int32_t>(static_cast<size_t>(rows * 2), value));
      test.AddOutput<int32_t>("Y", {rows, 1}, std::vector<int32_t>(static_cast<size_t>(rows), value));
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuReductionTest, NaiveLogSumExpInfinities) {
  for (const float value : {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("ReduceLogSumExp", 13);
    test.AddAttribute<std::vector<int64_t>>("axes", {1});
    test.AddInput<float>("X", {1025, 2}, std::vector<float>(2050, value));
    test.AddOutput<float>("Y", {1025, 1}, std::vector<float>(1025, value));
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuReductionTest, UnchangedSharedReductionControls) {
  for (const std::string op : {"ReduceMax", "ReduceSum", "ReduceProd"}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test(op.c_str(), 13);
    if (op != "ReduceSum") test.AddAttribute<std::vector<int64_t>>("axes", {1});
    test.AddInput<float>("X", {2, 3}, {1, 5, 3, -7, -2, -6});
    if (op == "ReduceSum") test.AddInput<int64_t>("axes", {1}, {1});
    const std::vector<float> expected = op == "ReduceMax"   ? std::vector<float>{5, -2}
                                        : op == "ReduceSum" ? std::vector<float>{9, -15}
                                                            : std::vector<float>{15, -84};
    test.AddOutput<float>("Y", {2, 1}, expected);
    RunWebGpu(test, std::move(ep));
  }
}

#endif
}  // namespace onnxruntime::test
