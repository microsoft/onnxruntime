// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"
#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/nn/layer_norm.h"
#include "core/providers/webgpu/nn/instance_norm.h"
#include "core/providers/webgpu/program_cache_key.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {
using namespace webgpu;

struct OtherLayerNormShader : LayerNormShader {};

#define WEBGPU_TEST_METADATA_CONFIG(F)
struct MetadataShader {
  WEBGPU_DECLARE_CONFIG(Config, WEBGPU_TEST_METADATA_CONFIG);
  static constexpr std::string_view name = "MetadataTest";
  static Status GenerateShaderCode(const Config&, ConfiguredShaderHelper&) { return Status::OK(); }
};
#undef WEBGPU_TEST_METADATA_CONFIG
using MetadataProgram = ConfiguredProgram<MetadataShader>;

std::string Key(const ProgramBase& program) {
  std::vector<uint32_t> inputs(program.Inputs().size(), 1);
  std::vector<uint32_t> outputs(program.Outputs().size(), 1);
  return CalculateProgramCacheKey(program, inputs, outputs);
}

TEST(ConfiguredProgramTest, EveryDeclaredFlagChangesIdentity) {
  LayerNormProgram baseline{false, false, false, false, false, false};
  ConfiguredProgram<OtherLayerNormShader> same_name{false, false, false, false, false, false};
  EXPECT_NE(Key(baseline), Key(same_name));
  InlinedHashSet<std::string> keys{Key(baseline)};
  for (size_t bit = 0; bit < 6; ++bit) {
    std::array<bool, 6> flags{};
    flags[bit] = true;
    LayerNormProgram variant{flags[0], flags[1], flags[2], flags[3], flags[4], flags[5]};
    EXPECT_TRUE(keys.insert(Key(variant)).second);
  }
  ComputeChannelScaleShiftProgram low_epsilon{4, 1e-5f, 64};
  ComputeChannelScaleShiftProgram high_epsilon{4, 1.0f, 64};
  EXPECT_NE(Key(low_epsilon), Key(high_epsilon));
}

TEST(ConfiguredProgramTest, TypesAndWidthsAreAutomaticAndUniformValuesAreNotKeyed) {
  std::array<float, 16> values{};
  OrtMemoryInfo memory{CPU, OrtDeviceAllocator};
  Tensor a{DataTypeImpl::GetType<float>(), TensorShape{4}, values.data(), memory};
  Tensor b{DataTypeImpl::GetType<float>(), TensorShape{8}, values.data(), memory};
  Tensor rank_two{DataTypeImpl::GetType<float>(), TensorShape{1, 4}, values.data(), memory};
  Tensor half{DataTypeImpl::GetType<MLFloat16>(), TensorShape{4}, values.data(), memory};
  auto make_key = [](Tensor& tensor, int components, uint32_t uniform) {
    MetadataProgram program;
    program.AddInput({&tensor, ProgramTensorMetadataDependency::None, components})
        .AddOutput({&tensor, ProgramTensorMetadataDependency::None, components})
        .AddUniformVariables({uniform, 1u});
    return Key(program);
  };
  EXPECT_EQ(make_key(a, 4, 1), make_key(b, 4, 2));
  EXPECT_EQ(make_key(a, 4, 1), make_key(a, 4, 99));
  EXPECT_NE(make_key(a, 4, 1), make_key(a, 2, 1));
  EXPECT_NE(make_key(a, 4, 1), make_key(half, 4, 1));
  EXPECT_NE(make_key(a, 4, 1), make_key(rank_two, 4, 1));
  const auto static_key = [](Tensor& tensor) {
    MetadataProgram program;
    program.AddInput({&tensor, ProgramTensorMetadataDependency::Shape});
    return Key(program);
  };
  EXPECT_NE(static_key(a), static_key(b));
  const auto offset_key = [&](uint32_t offset, bool uniform_offset) {
    MetadataProgram program;
    auto input = uniform_offset
                     ? ProgramInput::BufferView(&a, ProgramTensorMetadataDependency::None, TensorShape{1}, offset)
                     : ProgramInput{&a};
    input.buffer_offset_in_elements = offset;
    program.AddInput(std::move(input));
    return Key(program);
  };
  EXPECT_NE(offset_key(0, false), offset_key(1, false));
  EXPECT_EQ(offset_key(0, true), offset_key(1, true));
}

TEST(ConfiguredProgramTest, ExactKeysRemainDistinctWhenHashesCollide) {
  struct ConstantHash {
    size_t operator()(const std::string&) const { return 0; }
  };
  std::unordered_map<std::string, int, ConstantHash> cache;
  LayerNormProgram mean{false, false, true, false, false, false};
  LayerNormProgram inv_std{false, false, false, true, false, false};
  cache.emplace(Key(mean), 1);
  cache.emplace(Key(inv_std), 2);
  ASSERT_EQ(cache.size(), 2u);
  EXPECT_EQ(cache.at(Key(mean)), 1);
  EXPECT_EQ(cache.at(Key(inv_std)), 2);
  EXPECT_NE(Key(mean).find('\0'), std::string::npos);
  const auto display = ProgramCacheKeyForLogging(Key(mean));
  EXPECT_TRUE(display.starts_with("configured:"));
  EXPECT_EQ(display.find('\0'), std::string::npos);
}

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

TEST(ConfiguredProgramTest, OptionalOutputRolesExecuteCorrectlyInEitherOrder) {
  for (bool reverse : {false, true}) {
    auto provider = DefaultWebGpuExecutionProvider();
    if (!provider) GTEST_SKIP() << "WebGPU EP unavailable";
    LayerNormOutputsTester test{reverse};
    test.AddInput<float>("x", {1, 4}, {1, 2, 3, 4});
    test.AddInput<float>("scale", {4}, {1, 1, 1, 1});
    const float inv_std = 1.0f / std::sqrt(1.25f + 1e-5f);
    const std::vector<float> normalized{-1.5f * inv_std, -0.5f * inv_std, 0.5f * inv_std, 1.5f * inv_std};
    test.AddOutput<float>("mean_y", {1, 4}, normalized);
    test.AddOutput<float>("mean", {1, 1}, {2.5f});
    test.AddOutput<float>("inverse_y", {1, 4}, normalized);
    test.AddOutput<float>("inverse_std", {1, 1}, {inv_std});
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
  }
}

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
      auto& node =
          graph.AddNode(MakeString("variant", index), op_, "", node_inputs, {outputs[index]}, nullptr, domain_);
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

TEST(ConfiguredProgramTest, KnownCollisionsExecuteCorrectlyInEitherOrder) {
  for (bool reverse : {false, true}) {
    for (const char* op : {
#ifndef DISABLE_CONTRIB_OPS
             "BiasAdd", "BiasSplitGelu",
#endif
             "InstanceNormalization", "And"}) {
      SCOPED_TRACE(MakeString(op, " reverse=", reverse));
      ConfigOptions provider_options;
      ASSERT_STATUS_OK(provider_options.AddConfigEntry("ep.webgpuexecutionprovider.validationMode", "full"));
      ASSERT_STATUS_OK(provider_options.AddConfigEntry("ep.webgpuexecutionprovider.preferredLayout", "NCHW"));
      auto provider = WebGpuExecutionProviderWithOptions(provider_options);
      if (!provider) GTEST_SKIP() << "WebGPU EP unavailable";
      const std::string_view name{op};
      const bool bias = name == "BiasAdd" || name == "BiasSplitGelu";
      const size_t inputs_per_node = name == "BiasAdd" || name == "InstanceNormalization" ? 3 : 2;
      CachePairTester test{op, bias ? kMSDomain : kOnnxDomain, bias ? 1 : 17, inputs_per_node, reverse};
      if (bias) {
        const int64_t first_channels = name == "BiasAdd" ? 6 : 8;
        const int64_t first_output = name == "BiasAdd" ? first_channels : first_channels / 2;
        const int64_t second_output = name == "BiasAdd" ? 8 : 4;
        test.AddInput<float>("x0", {1, 1, first_channels}, std::vector<float>(first_channels, 1));
        test.AddInput<float>("b0", {first_channels}, std::vector<float>(first_channels, 2));
        if (name == "BiasAdd") {
          test.AddInput<float>("r0", {1, 1, first_channels}, std::vector<float>(first_channels, 3));
        }
        test.AddInput<MLFloat16>("x1", {1, 1, 8}, std::vector<MLFloat16>(8, MLFloat16{1.0f}));
        test.AddInput<MLFloat16>("b1", {8}, std::vector<MLFloat16>(8, MLFloat16{2.0f}));
        if (name == "BiasAdd") {
          test.AddInput<MLFloat16>("r1", {1, 1, 8}, std::vector<MLFloat16>(8, MLFloat16{3.0f}));
        }
        const float expected = name == "BiasAdd" ? 6.0f : 9.0f * 0.5f * (1.0f + std::erf(3.0f / std::sqrt(2.0f)));
        test.AddOutput<float>("y0", {1, 1, first_output}, std::vector<float>(first_output, expected));
        test.AddOutput<MLFloat16>("y1", {1, 1, second_output},
                                  std::vector<MLFloat16>(second_output, MLFloat16{expected}));
      } else if (name == "InstanceNormalization") {
        for (const char* suffix : {"0", "1"}) {
          test.AddInput<float>(MakeString("x", suffix).c_str(), {1, 1, 4}, {1, 2, 3, 4});
          test.AddInput<float>(MakeString("s", suffix).c_str(), {1}, {1});
          test.AddInput<float>(MakeString("b", suffix).c_str(), {1}, {0});
        }
        const float inv = 1.0f / std::sqrt(1.25f + 1e-5f);
        test.AddOutput<float>("y0", {1, 1, 4}, {-1.5f * inv, -0.5f * inv, 0.5f * inv, 1.5f * inv});
        test.AddOutput<float>("y1", {1, 1, 4}, {-1, -1.0f / 3, 1.0f / 3, 1});
      } else {
        test.AddInput<bool>("a0", {2, 1}, {true, false});
        test.AddInput<bool>("b0", {1, 4}, {true, false, true, false});
        test.AddInput<bool>("a1", {1, 4}, {true, false, true, false});
        test.AddInput<bool>("b1", {2, 1}, {true, false});
        const std::array<bool, 8> expected{true, false, true, false, false, false, false, false};
        test.AddOutput<bool>("y0", {2, 4}, expected.data(), expected.size());
        test.AddOutput<bool>("y1", {2, 4}, expected.data(), expected.size());
      }
      SessionOptions options;
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
      test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
    }
  }
}

TEST(ConfiguredProgramTest, VariableLengthFieldsHaveUnambiguousBoundaries) {
  const auto encode = [](const auto& value) {
    std::string key;
    AppendConfigValue(key, value);
    return key;
  };
  EXPECT_NE(encode(std::vector<std::string>{"ab", "c"}), encode(std::vector<std::string>{"a", "bc"}));
  EXPECT_NE(encode(std::vector<std::vector<int>>{{1}, {2, 3}}), encode(std::vector<std::vector<int>>{{1, 2}, {3}}));
  EXPECT_NE(encode(-1), encode(1));
  EXPECT_NE(encode(0.0f), encode(-0.0f));
  EXPECT_EQ(encode(std::string{"a\0b", 3}), encode(std::string_view{"a\0b", 3}));
  static_assert(!std::is_constructible_v<ShaderLiteral, std::string_view>);
  static_assert(!std::is_constructible_v<ShaderLiteral, char (&)[4]>);
  static constexpr char code[] = "return x;";
  EXPECT_EQ(encode(ShaderLiteral{code}), encode(ShaderLiteral{code}));
  EXPECT_NE(encode(ShaderLiteral{code}), encode(ShaderLiteral{"return y;"}));
  EXPECT_EQ(ShaderLiteral{code}.Text(), "return x;");
  constexpr ShaderLiteral long_code{
      "fn example() { let x = 123456789; let y = 123456789; let z = 123456789; return x + y + z; }"};
  EXPECT_LT(encode(long_code).size(), 32u);
}

}  // namespace
}  // namespace onnxruntime::test
