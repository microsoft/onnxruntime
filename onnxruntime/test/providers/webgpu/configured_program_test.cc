// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"
#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/nn/layer_norm.h"
#include "core/providers/webgpu/nn/instance_norm.h"
#include "contrib_ops/webgpu/bert/bias_add.h"
#include "core/providers/webgpu/program_cache_key.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {
using namespace webgpu;

struct OtherLayerNormShader : LayerNormShader {};

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
  Tensor half{DataTypeImpl::GetType<MLFloat16>(), TensorShape{4}, values.data(), memory};
  auto make_key = [](Tensor& tensor, int components, uint32_t uniform) {
    contrib::webgpu::BiasAddProgram program;
    program.AddInput({&tensor, ProgramTensorMetadataDependency::None, components})
        .AddOutput({&tensor, ProgramTensorMetadataDependency::None, components})
        .AddUniformVariables({uniform, 1u});
    return Key(program);
  };
  EXPECT_EQ(make_key(a, 4, 1), make_key(b, 4, 2));
  EXPECT_EQ(make_key(a, 4, 1), make_key(a, 4, 99));
  EXPECT_NE(make_key(a, 4, 1), make_key(a, 2, 1));
  EXPECT_NE(make_key(a, 4, 1), make_key(half, 4, 1));
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

class LegacyBenchmarkProgram final : public Program<LegacyBenchmarkProgram> {
 public:
  explicit LegacyBenchmarkProgram(std::string_view name) : Program{name} {}
  Status GenerateShaderCode(ShaderHelper&) const override { return Status::OK(); }
};

// Manually selected benchmark, excluded from CI. Includes per-dispatch configuration,
// framework metadata construction, key creation, hashing and a warm exact map lookup.
TEST(ConfiguredProgramTest, DISABLED_WarmKeyBenchmark) {
  std::array<float, 1024> values{};
  OrtMemoryInfo memory{CPU, OrtDeviceAllocator};
  Tensor x{DataTypeImpl::GetType<float>(), TensorShape{1, 1024}, values.data(), memory};
  Tensor scale{DataTypeImpl::GetType<float>(), TensorShape{1024}, values.data(), memory};
  Tensor y{DataTypeImpl::GetType<float>(), TensorShape{1, 1024}, values.data(), memory};
  const auto configure = [&](ProgramBase& program, int sample) {
    if (sample == 0) {
      program.AddInputs({{&x, ProgramTensorMetadataDependency::None, 4},
                         {&scale, ProgramTensorMetadataDependency::None, 4},
                         {&x, ProgramTensorMetadataDependency::None, 4}})
          .AddOutput({&y, ProgramTensorMetadataDependency::None, 4})
          .AddUniformVariables({256u, 256u});
      return;
    }
    if (sample == 2) {
      program.AddInputs({{&x, ProgramTensorMetadataDependency::TypeAndRank, TensorShape{1, 1, 256}, 4},
                         {&scale, ProgramTensorMetadataDependency::TypeAndRank},
                         {&scale, ProgramTensorMetadataDependency::TypeAndRank}})
          .AddOutput({&y, ProgramTensorMetadataDependency::TypeAndRank, TensorShape{1, 1, 1}, 2})
          .SetWorkgroupSize(64);
      return;
    }
    program.AddInput({&x, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4});
    program.AddInput({&scale, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4});
    program.AddOutput({&y, ProgramTensorMetadataDependency::Type, ProgramOutput::Flatten, 4});
    program.AddUniformVariables({4u, 1u, 1024u, 256u, 1e-5f});
  };
  constexpr size_t iterations = 100000;
  for (int sample : {0, 1, 2}) {
    const char* name = sample == 0 ? "BiasAdd" : sample == 1 ? "LayerNorm"
                                                             : "ComputeChannelScaleShift";
    const auto make_key = [&](bool configured) {
      if (!configured) {
        LegacyBenchmarkProgram program{name};
        if (sample == 1) program.CacheHint(4, false, false, false);
        if (sample == 2) program.CacheHint(4, 1);
        configure(program, sample);
        return Key(program);
      }
      if (sample == 0) {
        contrib::webgpu::BiasAddProgram program;
        configure(program, sample);
        return Key(program);
      }
      if (sample == 1) {
        LayerNormProgram program{false, false, false, false, false, false};
        configure(program, sample);
        return Key(program);
      }
      ComputeChannelScaleShiftProgram program{4, 1e-5f, 64};
      configure(program, sample);
      return Key(program);
    };
    std::unordered_map<std::string, size_t> cache{{make_key(false), 1}, {make_key(true), 1}};
    for (int trial = 0; trial < 12; ++trial) {
      for (bool configured : {trial % 2 == 0, trial % 2 != 0}) {
        size_t checksum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iterations; ++i) checksum += cache.at(make_key(configured));
        const auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        ASSERT_EQ(checksum, iterations);
        std::cout << "KEY_BENCH," << name << ',' << trial << ',' << configured << ','
                  << ns / iterations << ',' << make_key(configured).size() << '\n';
      }
    }
  }
}

}  // namespace
}  // namespace onnxruntime::test
