// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <functional>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/graph/graph.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

// Keep both variants in one session so they share the ready/pending shader cache.
class CachePairTester final : public OpTester {
 public:
  CachePairTester(const char* op, const char* domain, int version, size_t input_count, bool reverse)
      : OpTester(op, version, domain), op_{op}, domain_{domain}, input_count_{input_count}, reverse_{reverse} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CachePairTester);

  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>&) override {
    for (size_t step = 0; step < 2; ++step) {
      const size_t index = reverse_ ? 1 - step : step;
      std::vector<NodeArg*> node_inputs(inputs.begin() + index * input_count_,
                                        inputs.begin() + (index + 1) * input_count_);
      auto& node = graph.AddNode(MakeString("variant", index), op_, "", node_inputs, {outputs[index]}, nullptr, domain_);
      if (std::string_view{op_} == "InstanceNormalization") {
        node.AddAttribute("epsilon", index == 0 ? 1e-5f : 1.0f);
      }
    }
  }

 private:
  const char* op_;
  const char* domain_;
  const size_t input_count_;
  const bool reverse_;
};

void RunOnWebGpu(OpTester& test) {
  ConfigOptions provider_options;
  ASSERT_STATUS_OK(provider_options.AddConfigEntry("ep.webgpuexecutionprovider.validationMode", "full"));
  ASSERT_STATUS_OK(provider_options.AddConfigEntry("ep.webgpuexecutionprovider.preferredLayout", "NCHW"));
  auto provider = WebGpuExecutionProviderWithOptions(provider_options);
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.SetNumRunCalls(2);
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}

#ifndef DISABLE_CONTRIB_OPS
template <typename T>
void AddBiasVariant(CachePairTester& test, const char* op, int index, int64_t channels) {
  const bool add = std::string_view{op} == "BiasAdd";
  const int64_t output_channels = add ? channels : channels / 2;
  test.AddInput<T>(MakeString("x", index).c_str(), {1, 1, channels}, std::vector<T>(channels, T{1.0f}));
  test.AddInput<T>(MakeString("b", index).c_str(), {channels}, std::vector<T>(channels, T{2.0f}));
  if (add) {
    test.AddInput<T>(MakeString("r", index).c_str(), {1, 1, channels}, std::vector<T>(channels, T{3.0f}));
  }
  const float expected = add ? 6.0f : 4.5f * (1.0f + std::erf(3.0f / std::sqrt(2.0f)));
  test.AddOutput<T>(MakeString("y", index).c_str(), {1, 1, output_channels},
                    std::vector<T>(output_channels, T{expected}));
}

TEST(WebGpuShaderCacheTest, BiasAddVectorWidths) {
  for (bool reverse : {false, true}) {
    // Scalar and vec2 must each coexist with vec4 at the same rank and dtype.
    for (int64_t channels : {3, 6}) {
      SCOPED_TRACE(MakeString("reverse=", reverse, " channels=", channels));
      CachePairTester test{"BiasAdd", kMSDomain, 1, 3, reverse};
      AddBiasVariant<float>(test, "BiasAdd", 0, channels);
      AddBiasVariant<float>(test, "BiasAdd", 1, 8);
      RunOnWebGpu(test);
    }
  }
}

void TestBiasElementTypes(const char* op) {
  for (bool reverse : {false, true}) {
    SCOPED_TRACE(MakeString(op, " reverse=", reverse));
    CachePairTester test{op, kMSDomain, 1, std::string_view{op} == "BiasAdd" ? 3u : 2u, reverse};
    AddBiasVariant<float>(test, op, 0, 8);
    AddBiasVariant<MLFloat16>(test, op, 1, 8);
    RunOnWebGpu(test);
  }
}

TEST(WebGpuShaderCacheTest, BiasAddElementTypes) {
  TestBiasElementTypes("BiasAdd");
}

TEST(WebGpuShaderCacheTest, BiasSplitGeluElementTypes) {
  TestBiasElementTypes("BiasSplitGelu");
}
#endif

class LayerNormOutputsTester final : public OpTester {
 public:
  explicit LayerNormOutputsTester(bool reverse) : OpTester("LayerNormalization", 17), reverse_{reverse} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LayerNormOutputsTester);

  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>&) override {
    auto& absent = graph.GetOrCreateNodeArg("", nullptr);
    for (int i : {0, 1}) {
      const bool mean = (i == 0) != reverse_;
      graph.AddNode(mean ? "mean" : "inverse_std", "LayerNormalization", "", inputs,
                    mean ? std::vector<NodeArg*>{outputs[0], outputs[1]}
                         : std::vector<NodeArg*>{outputs[2], &absent, outputs[3]});
    }
  }

 private:
  const bool reverse_;
};

TEST(WebGpuShaderCacheTest, LayerNormOptionalOutputRoles) {
  for (bool reverse : {false, true}) {
    SCOPED_TRACE(reverse);
    LayerNormOutputsTester test{reverse};
    test.AddInput<float>("x", {1, 4}, {1, 2, 3, 4});
    test.AddInput<float>("scale", {4}, {1, 1, 1, 1});
    const float inv_std = 1.0f / std::sqrt(1.25f + 1e-5f);
    const std::vector<float> normalized{-1.5f * inv_std, -0.5f * inv_std, 0.5f * inv_std, 1.5f * inv_std};
    test.AddOutput<float>("mean_y", {1, 4}, normalized);
    test.AddOutput<float>("mean", {1, 1}, {2.5f});
    test.AddOutput<float>("inverse_y", {1, 4}, normalized);
    test.AddOutput<float>("inverse_std", {1, 1}, {inv_std});
    RunOnWebGpu(test);
  }
}

TEST(WebGpuShaderCacheTest, InstanceNormEpsilon) {
  for (bool reverse : {false, true}) {
    SCOPED_TRACE(reverse);
    CachePairTester test{"InstanceNormalization", kOnnxDomain, 17, 3, reverse};
    for (int index : {0, 1}) {
      test.AddInput<float>(MakeString("x", index).c_str(), {1, 1, 4}, {1, 2, 3, 4});
      test.AddInput<float>(MakeString("s", index).c_str(), {1}, {1});
      test.AddInput<float>(MakeString("b", index).c_str(), {1}, {0});
    }
    const float inv = 1.0f / std::sqrt(1.25f + 1e-5f);
    test.AddOutput<float>("y0", {1, 1, 4}, {-1.5f * inv, -0.5f * inv, 0.5f * inv, 1.5f * inv});
    test.AddOutput<float>("y1", {1, 1, 4}, {-1, -1.0f / 3, 1.0f / 3, 1});
    RunOnWebGpu(test);
  }
}

TEST(WebGpuShaderCacheTest, BooleanBroadcastReadModes) {
  for (bool reverse : {false, true}) {
    SCOPED_TRACE(reverse);
    CachePairTester test{"And", kOnnxDomain, 17, 2, reverse};
    test.AddInput<bool>("a0", {2, 1}, {true, false});
    test.AddInput<bool>("b0", {1, 4}, {true, false, true, false});
    test.AddInput<bool>("a1", {1, 4}, {true, false, true, false});
    test.AddInput<bool>("b1", {2, 1}, {true, false});
    const std::array<bool, 8> expected{true, false, true, false, false, false, false, false};
    test.AddOutput<bool>("y0", {2, 4}, expected.data(), expected.size());
    test.AddOutput<bool>("y1", {2, 4}, expected.data(), expected.size());
    RunOnWebGpu(test);
  }
}

}  // namespace
}  // namespace onnxruntime::test
