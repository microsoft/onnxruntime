// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/shader_config.h"
#include <type_traits>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_variable.h"

namespace onnxruntime::webgpu {

// No ProgramBase, Tensor, Context or uniform payload accessor is exposed to generators.
// Only keyed metadata is observable. Shape expressions use uniforms unless the
// host explicitly selects keyed static dimensions.
class ConfiguredShaderHelper final {
 public:
  explicit ConfiguredShaderHelper(ShaderHelper& shader) : shader_{shader} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConfiguredShaderHelper);

  const ShaderVariableHelper& AddInput(const std::string& name,
                                       ShaderUsage usage = ShaderUsage::UseIndicesTypeAlias |
                                                           ShaderUsage::UseValueTypeAlias);

  const ShaderVariableHelper& AddOutput(const std::string& name,
                                        ShaderUsage usage = ShaderUsage::UseIndicesTypeAlias |
                                                            ShaderUsage::UseValueTypeAlias);

  const ShaderIndicesHelper& AddIndices(const std::string& name,
                                        ShaderUsage usage = ShaderUsage::None);

  size_t InputCount() const;
  size_t OutputCount() const;
  uint32_t WorkgroupSizeX() const;
  uint32_t WorkgroupSizeY() const;
  uint32_t WorkgroupSizeZ() const;
  int InputElementType(size_t index) const;
  int OutputElementType(size_t index) const;
  ProgramVariableDataType InputType(size_t index) const;
  ProgramVariableDataType OutputType(size_t index) const;
  OStringStream& AdditionalImplementation();
  OStringStream& MainFunctionBody();
  std::string GuardAgainstOutOfBoundsWorkgroupSizes(std::string_view size) const;

 private:
  ShaderHelper& shader_;
};

// Spec is never instantiated: generation has no receiver carrying uncaptured members.
template <typename Spec>
class ConfiguredProgram final : public Program<Spec> {
 public:
  using Config = typename Spec::Config;
  template <typename... Args>
  explicit ConfiguredProgram(Args&&... args) : ConfiguredProgram(Config(std::forward<Args>(args)...), 0) {}
  ConfiguredProgram(ConfiguredProgram&&) = default;
  ConfiguredProgram& operator=(ConfiguredProgram&&) = delete;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ConfiguredProgram);

  const Config& Specialization() const { return config_; }

  Status GenerateShaderCode(ShaderHelper& shader) const final {
    ConfiguredShaderHelper view{shader};
    return Spec::GenerateShaderCode(config_, view);
  }

  const void* StructuredKeyType() const final { return &type_token_; }
  void AppendSpecializationKey(std::string& key) const final { config_.AppendTo(key); }

 private:
  static std::string_view DisplayName(const Config& config) {
    if constexpr (requires { Spec::Name(config); })
      return Spec::Name(config);
    else
      return Spec::name;
  }
  ConfiguredProgram(Config config, int) : Program<Spec>{DisplayName(config)}, config_{std::move(config)} {
    static_assert(std::is_same_v<typename Config::ShaderConfigSchema, void>);
    static_assert(std::is_final_v<Config>);
    static_assert(std::is_empty_v<Spec>, "Generator specifications must not contain instance state");
    static_assert(
        std::is_same_v<decltype(&Spec::GenerateShaderCode), Status (*)(const Config&, ConfiguredShaderHelper&)>);
  }
  inline static char type_token_ = 0;
  Config config_;
};

}  // namespace onnxruntime::webgpu
