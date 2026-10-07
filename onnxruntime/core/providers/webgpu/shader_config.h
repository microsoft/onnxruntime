// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <bit>
#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "core/common/inlined_containers_fwd.h"
#include "core/common/common.h"
#include <gsl/gsl>

namespace onnxruntime::webgpu {

// A value reference to constant shader code. Immediate construction excludes
// runtime strings and automatic storage; copies retain the same immutable code.
class ShaderLiteral final {
 public:
  constexpr ShaderLiteral() = default;
  template <typename T, size_t N>
    requires std::is_same_v<T, const char>
  consteval ShaderLiteral(T (&text)[N]) : text_{N == 1 ? std::string_view{} : std::string_view{text, N - 1}} {
    ORT_ENFORCE(text[N - 1] == '\0', "Shader literals must be null terminated");
  }

  std::string_view Text() const { return text_; }

 private:
  template <typename T>
  friend void AppendConfigValue(std::string&, const T&);
  std::string_view text_;
};

namespace detail {
template <typename T>
inline constexpr bool is_config_sequence = false;
template <typename T, typename A>
inline constexpr bool is_config_sequence<std::vector<T, A>> = true;
#ifndef DISABLE_ABSEIL
template <typename T, size_t N, typename A>
inline constexpr bool is_config_sequence<InlinedVector<T, N, A>> = true;
#endif
template <typename T, size_t N>
inline constexpr bool is_config_sequence<std::array<T, N>> = true;
template <typename T, size_t N>
inline constexpr bool is_config_sequence<std::span<T, N>> = true;
template <typename T, size_t N>
inline constexpr bool is_config_sequence<gsl::span<T, N>> = true;
template <typename K, typename V, typename C, typename A>
inline constexpr bool is_config_sequence<std::map<K, V, C, A>> = true;
template <typename T>
inline constexpr bool is_config_pair = false;
template <typename A, typename B>
inline constexpr bool is_config_pair<std::pair<A, B>> = true;
}  // namespace detail

// The key owns the encoded values. No padding bytes, addresses or lossy digests
// participate in equality; collection lengths delimit variable-sized fields.
template <typename T>
void AppendConfigValue(std::string& key, const T& value) {
  if constexpr (std::is_same_v<T, ShaderLiteral>) {
    // Like the generator token, this identifies immutable code, not tensor data.
    AppendConfigValue(key, reinterpret_cast<uintptr_t>(value.text_.data()));
    AppendConfigValue(key, value.text_.size());
  } else if constexpr (std::is_same_v<T, float>) {
    AppendConfigValue(key, std::bit_cast<uint32_t>(value));
  } else if constexpr (std::is_same_v<T, double>) {
    AppendConfigValue(key, std::bit_cast<uint64_t>(value));
  } else if constexpr (std::is_enum_v<T>) {
    AppendConfigValue(key, static_cast<std::underlying_type_t<T>>(value));
  } else if constexpr (std::is_integral_v<T>) {
    static_assert(sizeof(T) <= sizeof(uint64_t));
    uint64_t remaining = static_cast<uint64_t>(value);
    while (remaining >= 128) {
      key.push_back(static_cast<char>((remaining & 127) | 128));
      remaining >>= 7;
    }
    key.push_back(static_cast<char>(remaining));
  } else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>) {
    AppendConfigValue(key, value.size());
    key.append(value);
  } else if constexpr (requires { typename T::ShaderConfigSchema; }) {
    value.AppendTo(key);
  } else if constexpr (detail::is_config_pair<T>) {
    AppendConfigValue(key, value.first);
    AppendConfigValue(key, value.second);
  } else if constexpr (detail::is_config_sequence<T>) {
    AppendConfigValue(key, value.size());
    for (const auto& item : value) AppendConfigValue(key, item);
  } else {
    static_assert(sizeof(T) == 0, "Shader configuration needs a declared field schema; pointers are not supported");
  }
}

#define WEBGPU_CONFIG_FIELD(type, name) type name{};
#define WEBGPU_CONFIG_ENCODE(type, name) onnxruntime::webgpu::AppendConfigValue(key, name);
#define WEBGPU_CONFIG_MEMBERS(fields) \
  using ShaderConfigSchema = void;    \
  fields(WEBGPU_CONFIG_FIELD) void AppendTo([[maybe_unused]] std::string& key) const { fields(WEBGPU_CONFIG_ENCODE) }
#define WEBGPU_DECLARE_CONFIG(name, fields) \
  struct name final {                       \
    WEBGPU_CONFIG_MEMBERS(fields)           \
  }

}  // namespace onnxruntime::webgpu
