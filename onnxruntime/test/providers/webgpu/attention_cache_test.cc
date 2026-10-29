// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)
namespace {
std::unique_ptr<IExecutionProvider> CreateWebGpuProvider() {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.validationMode", "full"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.storageBufferCacheMode", "disabled"));
  return WebGpuExecutionProviderWithOptions(options);
}
void RunWebGpu(OpTester& test, std::unique_ptr<IExecutionProvider> provider) {
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  options.graph_optimization_level = TransformerLevel::Default;
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}
}  // namespace

TEST(WebGpuAttentionCacheTest, MultiHeadAttentionDifferentValueHeadSize) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("MultiHeadAttention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 1);
  test.AddInput<float>("query", {1, 1, 2}, {0, 0});
  test.AddInput<float>("key", {1, 1, 2}, {0, 0});
  test.AddInput<float>("value", {1, 1, 4}, {1, 2, 3, 4});
  test.AddOutput<float>("output", {1, 1, 4}, {1, 2, 3, 4});
  test.AddOutput<float>("present_key", {1, 1, 1, 2}, {0, 0});
  test.AddOutput<float>("present_value", {1, 1, 1, 4}, {1, 2, 3, 4});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuAttentionCacheTest, MultiHeadAttentionFirstPromptCaches) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("MultiHeadAttention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 1);
  test.AddInput<float>("query", {1, 1, 2}, {0, 0});
  test.AddInput<float>("key", {1, 1, 2}, {2, 3});
  test.AddInput<float>("value", {1, 1, 2}, {4, 5});
  test.AddOutput<float>("output", {1, 1, 2}, {4, 5});
  test.AddOutput<float>("present_key", {1, 1, 1, 2}, {2, 3});
  test.AddOutput<float>("present_value", {1, 1, 1, 2}, {4, 5});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuAttentionCacheTest, MultiHeadAttentionDecodeOptionalCaches) {
  for (const int outputs : {0, 1, 2, 3}) {
    SCOPED_TRACE(outputs);
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("MultiHeadAttention", 1, kMSDomain);
    test.AddAttribute<int64_t>("num_heads", 1);
    test.AddInput<float>("query", {1, 1, 2}, {0, 0});
    test.AddInput<float>("key", {1, 1, 2}, {2, 3});
    test.AddInput<float>("value", {1, 1, 2}, {4, 6});
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<int32_t>();
    test.AddOptionalInputEdge<float>();
    test.AddInput<float>("past_key", {1, 1, 1, 2}, {1, 1});
    test.AddInput<float>("past_value", {1, 1, 1, 2}, {2, 4});
    // MHA ignores past inputs when neither cache output is requested, matching CPU.
    test.AddOutput<float>("output", {1, 1, 2}, outputs == 0 ? std::vector<float>{4, 6} : std::vector<float>{3, 5});
    if ((outputs & 1) != 0) {
      test.AddOutput<float>("present_key", {1, 1, 2, 2}, {1, 1, 2, 3});
    } else if ((outputs & 2) != 0) {
      test.AddOptionalOutputEdge<float>();
    }
    if ((outputs & 2) != 0) {
      test.AddOutput<float>("present_value", {1, 1, 2, 2}, {2, 4, 4, 6});
    }
    RunWebGpu(test, std::move(ep));
  }
}
#endif
}  // namespace onnxruntime::test
