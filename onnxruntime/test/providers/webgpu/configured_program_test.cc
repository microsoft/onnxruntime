// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"
#include "core/common/inlined_containers.h"
#include "core/framework/tensor.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_metadata.h"
#include "core/providers/webgpu/program_cache_key.h"

namespace onnxruntime::test {
namespace {
using namespace webgpu;

#define WEBGPU_TEST_FIELDS(F) F(bool, flag0) F(bool, flag1) F(bool, flag2) F(bool, flag3)
struct TestShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TEST_FIELDS);
  static constexpr std::string_view name = "Test";
  static Status GenerateShaderCode(const Config&, ConfiguredShaderHelper&) { return Status::OK(); }
};
#undef WEBGPU_TEST_FIELDS
using TestProgram = ConfiguredProgram<TestShader>;
struct OtherTestShader : TestShader {};

#define WEBGPU_TEST_FLOAT_FIELDS(F) F(float, epsilon)
struct FloatShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TEST_FLOAT_FIELDS);
  static constexpr std::string_view name = "FloatTest";
  static Status GenerateShaderCode(const Config&, ConfiguredShaderHelper&) { return Status::OK(); }
};
#undef WEBGPU_TEST_FLOAT_FIELDS

#define WEBGPU_TEST_METADATA_CONFIG(F)
struct MetadataShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TEST_METADATA_CONFIG);
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
  TestProgram baseline{false, false, false, false};
  ConfiguredProgram<OtherTestShader> same_name{false, false, false, false};
  EXPECT_NE(Key(baseline), Key(same_name));
  InlinedHashSet<std::string> keys{Key(baseline)};
  for (size_t bit = 0; bit < 4; ++bit) {
    std::array<bool, 4> flags{};
    flags[bit] = true;
    TestProgram variant{flags[0], flags[1], flags[2], flags[3]};
    EXPECT_TRUE(keys.insert(Key(variant)).second);
  }
  ConfiguredProgram<FloatShader> low_epsilon{1e-5f};
  ConfiguredProgram<FloatShader> high_epsilon{1.0f};
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
  TestProgram mean{false, false, true, false};
  TestProgram inv_std{false, false, false, true};
  cache.emplace(Key(mean), 1);
  cache.emplace(Key(inv_std), 2);
  ASSERT_EQ(cache.size(), 2u);
  EXPECT_EQ(cache.at(Key(mean)), 1);
  EXPECT_EQ(cache.at(Key(inv_std)), 2);
  EXPECT_NE(Key(mean).find('\0'), std::string::npos);
  ShaderDiagnosticIds diagnostics;
  ShaderDiagnosticIds other_cache;
  const auto display = diagnostics.Format(Key(mean));
  EXPECT_TRUE(display.starts_with("shader:"));
  EXPECT_EQ(display.find_first_not_of("0123456789", 7), std::string::npos);
  EXPECT_EQ(display, diagnostics.Format(Key(mean)));
  EXPECT_NE(display, diagnostics.Format(Key(inv_std)));
  EXPECT_NE(display, other_cache.Format(Key(mean)));
  EXPECT_EQ(display.find('\0'), std::string::npos);
}

TEST(ConfiguredProgramTest, CommonDefaultsAndUniformExpressions) {
  MetadataProgram defaults;
  MetadataProgram explicit_defaults;
  explicit_defaults.SetWorkgroupSize(uint32_t(WORKGROUP_SIZE), 1, 1);
  EXPECT_EQ(Key(defaults), Key(explicit_defaults));
  EXPECT_EQ(ShaderProgramMetadata{defaults}.WorkgroupSizeX(), uint32_t(WORKGROUP_SIZE));
  using Uniform = ShaderUniform<ProgramUniformVariableDataType::Float32>;
  using Expression = ShaderUniformExpression<ProgramUniformVariableDataType::Float32>;
  static_assert(!std::is_convertible_v<Expression, float>);
  static_assert(!std::is_convertible_v<Expression, bool>);
  EXPECT_EQ(Expression{Uniform{"epsilon"}}.Expr(), "uniforms.epsilon");
}

#define WEBGPU_NAMED_TEST_FIELDS(F) F(bool, fast) F(uint32_t, tile)
struct NamedTestShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_NAMED_TEST_FIELDS);
  static ShaderProgramSetup<Config> MakeConfig(std::string_view name, bool fast, uint32_t tile) {
    return {{fast, fast ? tile : 0u}, name};
  }
  static Status GenerateShaderCode(const Config&, ConfiguredShaderHelper&) { return Status::OK(); }
};
#undef WEBGPU_NAMED_TEST_FIELDS

TEST(ConfiguredProgramTest, HostLabelsAndInactiveFieldsDoNotForkIdentity) {
  ConfiguredProgram<NamedTestShader> a{"first", false, 4};
  ConfiguredProgram<NamedTestShader> b{"second", false, 8};
  EXPECT_EQ(Key(a), Key(b));
  EXPECT_NE(a.Name(), b.Name());
  ConfiguredProgram<NamedTestShader> fast4{"first", true, 4};
  ConfiguredProgram<NamedTestShader> fast8{"first", true, 8};
  EXPECT_NE(Key(fast4), Key(fast8));
  static_assert(std::is_final_v<NamedTestShader::Config>);
  static_assert(std::is_aggregate_v<NamedTestShader::Config>);
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
