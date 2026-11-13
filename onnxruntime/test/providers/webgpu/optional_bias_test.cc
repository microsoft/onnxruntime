// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

enum class BiasInput { Omitted,
                       EmptySlot,
                       Present };

template <typename T>
void AddBiasInput(OpTester& test, BiasInput bias, const std::vector<T>& values, bool initializer = false) {
  if (bias == BiasInput::EmptySlot) {
    test.AddOptionalInputEdge<T>();
  } else if (bias == BiasInput::Present) {
    test.AddInput<T>("bias", {static_cast<int64_t>(values.size())}, values, initializer);
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

template <typename T>
void RunConvolution(OpTester& test, const std::vector<int64_t>& output_shape, float expected) {
  test.AddOutput<T>("Y", output_shape,
                    std::vector<T>(static_cast<size_t>(TensorShape(output_shape).Size()), T(expected)));
  test.SetCustomOutputVerifier([output_shape, expected](const std::vector<OrtValue>& fetches,
                                                        const std::string& provider_type) {
    ASSERT_EQ(provider_type, kWebGpuExecutionProvider);
    ::testing::Test::RecordProperty(provider_type, "executed");
    ASSERT_EQ(fetches.size(), 1u);
    const auto& output = fetches[0].Get<Tensor>();
    ASSERT_EQ(output.Shape(), TensorShape(output_shape));
    for (const auto& value : output.DataAsSpan<T>()) {
      EXPECT_EQ(value, T(expected));
    }
  });
  auto provider = CreateProvider("NHWC");
  if (!provider) GTEST_SKIP() << "WebGPU is not available";
  RunWebGpu(test, std::move(provider));
}

void TestPointwiseConvolution(const char* op) {
  for (const size_t rank : {4u, 5u}) {
    for (const bool initializer : {false, true}) {
      for (const auto bias : {BiasInput::Omitted, BiasInput::EmptySlot, BiasInput::Present}) {
        SCOPED_TRACE(testing::Message() << "rank=" << rank << " initializer=" << initializer
                                        << " bias=" << static_cast<int>(bias));
        OpTester test(op, 11);
        const std::vector<int64_t> shape(rank, 1);
        test.AddInput<float>("X", shape, {2});
        test.AddInput<float>("W", shape, {3}, initializer);
        AddBiasInput<float>(test, bias, {5.f}, initializer);
        RunConvolution<float>(test, shape, bias == BiasInput::Present ? 11.f : 6.f);
      }
    }
  }
}

template <typename T>
void TestSpatialConvolution() {
  for (const bool initializer : {false, true}) {
    for (const auto bias : {BiasInput::Omitted, BiasInput::EmptySlot, BiasInput::Present}) {
      SCOPED_TRACE(testing::Message() << "initializer=" << initializer << " bias=" << static_cast<int>(bias));
      OpTester test("Conv", 11);
      test.AddInput<T>("X", {1, 1, 3, 3}, std::vector<T>(9, T(2.f)));
      test.AddInput<T>("W", {1, 1, 2, 2}, std::vector<T>(4, T(3.f)), initializer);
      AddBiasInput<T>(test, bias, {T(5.f)}, initializer);
      RunConvolution<T>(test, {1, 1, 2, 2}, bias == BiasInput::Present ? 29.f : 24.f);
    }
  }
}

}  // namespace

TEST(WebGpuOptionalBiasTest, Conv2DAnd3D) {
  TestPointwiseConvolution("Conv");
}

TEST(WebGpuOptionalBiasTest, ConvTranspose2DAnd3D) {
  TestPointwiseConvolution("ConvTranspose");
}

TEST(WebGpuOptionalBiasTest, ConvSpatial) {
  TestSpatialConvolution<float>();
}

TEST(WebGpuOptionalBiasTest, ConvSpatialHalf) {
  TestSpatialConvolution<MLFloat16>();
}

#ifndef DISABLE_CONTRIB_OPS
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
