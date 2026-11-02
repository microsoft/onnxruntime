// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#ifndef DISABLE_CONTRIB_OPS
namespace {

enum class BiasInput { Omitted,
                       EmptySlot,
                       Present };

template <typename T>
void AddBiasInput(OpTester& test, BiasInput bias, const std::vector<T>& values) {
  if (bias == BiasInput::EmptySlot) {
    test.AddOptionalInputEdge<T>();
  } else if (bias == BiasInput::Present) {
    test.AddInput<T>("bias", {static_cast<int64_t>(values.size())}, values);
  }
}

std::unique_ptr<IExecutionProvider> CreateProvider(const char* layout) {
  ConfigOptions config;
  ORT_THROW_IF_ERROR(config.AddConfigEntry("ep.webgpuexecutionprovider.validationMode", "full"));
  ORT_THROW_IF_ERROR(config.AddConfigEntry("ep.webgpuexecutionprovider.preferredLayout", layout));
  return WebGpuExecutionProviderWithOptions(config);
}

void RunWebGpu(OpTester& test, std::unique_ptr<IExecutionProvider> provider) {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}

}  // namespace

TEST(WebGpuOptionalBiasTest, Attention) {
  for (const auto bias : {BiasInput::Omitted, BiasInput::EmptySlot, BiasInput::Present}) {
    SCOPED_TRACE(static_cast<int>(bias));
    auto provider = CreateProvider("NHWC");
    if (!provider) GTEST_SKIP() << "WebGPU is not available";
    OpTester test("Attention", 1, kMSDomain);
    test.AddAttribute<int64_t>("num_heads", 1);
    test.AddInput<float>("input", {1, 1, 2}, {2, 3});
    // Q and K are zero; the value projection preserves the input.
    test.AddInput<float>("weights", {2, 6}, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1});
    AddBiasInput<float>(test, bias, {0, 0, 0, 0, 5, 7});
    test.AddOutput<float>("output", {1, 1, 2},
                          bias == BiasInput::Present ? std::vector<float>{7, 10} : std::vector<float>{2, 3});
    RunWebGpu(test, std::move(provider));
  }
}
#endif

}  // namespace onnxruntime::test
