// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <bit>
#include <type_traits>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_variable.h"

namespace onnxruntime::webgpu {

// Exact scalar encoding, not a digest. Pointer/object representations are not supported.
template <typename T>
inline void AppendConfigScalar(std::string& key, T value) {
  static_assert(std::is_integral_v<T> || std::is_enum_v<T> || std::is_same_v<T, float>);
  if constexpr (std::is_same_v<T, float>) {
    AppendConfigScalar(key, std::bit_cast<uint32_t>(value));
  } else if constexpr (std::is_enum_v<T>) {
    AppendConfigScalar(key, static_cast<std::underlying_type_t<T>>(value));
  } else {
    static_assert(sizeof(T) <= sizeof(uint64_t));
    uint64_t remaining = static_cast<uint64_t>(value);
    while (remaining >= 128) {
      key.push_back(static_cast<char>((remaining & 127) | 128));
      remaining >>= 7;
    }
    key.push_back(static_cast<char>(remaining));
  }
}

#define WEBGPU_CONFIG_FIELD(type, name) const type name{};
#define WEBGPU_CONFIG_ENCODE(type, name) AppendConfigScalar(key, name);
#define WEBGPU_DECLARE_CONFIG(name, fields)                                              \
  struct name final {                                                                    \
    fields(WEBGPU_CONFIG_FIELD) void AppendTo([[maybe_unused]] std::string& key) const { \
      fields(WEBGPU_CONFIG_ENCODE)                                                       \
    }                                                                                    \
  }

// No ProgramBase, Tensor, Context or uniform payload accessor is exposed to generators.
// Shapes are always accessed through uniforms; only rank and shader type are observable.
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
  explicit ConfiguredProgram(Args&&... args)
      : Program<Spec>{Spec::name}, config_{std::forward<Args>(args)...} {
    static_assert(std::is_same_v<decltype(&Spec::GenerateShaderCode),
                                 Status (*)(const Config&, ConfiguredShaderHelper&)>);
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConfiguredProgram);

  Status GenerateShaderCode(ShaderHelper& shader) const final {
    ConfiguredShaderHelper view{shader};
    return Spec::GenerateShaderCode(config_, view);
  }

  const void* StructuredKeyType() const final { return &type_token_; }
  void AppendSpecializationKey(std::string& key) const final { config_.AppendTo(key); }

  template <typename... Args>
  void CacheHint(Args&&...) = delete;

 private:
  inline static char type_token_ = 0;
  const Config config_;
};

}  // namespace onnxruntime::webgpu
