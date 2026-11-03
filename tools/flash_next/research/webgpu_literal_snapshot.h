#pragma once

#include "core/platform/env_var_utils.h"
#include <string>
#include <type_traits>

namespace onnxruntime {

template <size_t N>
struct WebGpuLiteralFlagName {
  char value[N];
  constexpr WebGpuLiteralFlagName(const char (&text)[N]) {
    for (size_t i = 0; i < N; ++i) value[i] = text[i];
  }
};

inline bool WebGpuLiteralSnapshotsEnabled() {
  static const int enabled = [] {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_LITERAL_SNAPSHOT", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_LITERAL_SNAPSHOT must be 0 or 1");
    return value;
  }();
  return enabled == 1;
}

template <typename T, WebGpuLiteralFlagName Name>
optional<T> WebGpuLiteralSnapshotOptional() {
  if constexpr (!std::is_same_v<T, int>) {
    return ParseEnvironmentVariable<T>(std::string(Name.value));
  } else {
    if (!WebGpuLiteralSnapshotsEnabled()) {
      return ParseEnvironmentVariable<T>(std::string(Name.value));
    }
    static const optional<T> parsed = ParseEnvironmentVariable<T>(std::string(Name.value));
    return parsed;
  }
}

template <typename T, WebGpuLiteralFlagName Name>
T WebGpuLiteralSnapshotDefault(const T& fallback) {
  const auto parsed = WebGpuLiteralSnapshotOptional<T, Name>();
  return parsed.has_value() ? *parsed : fallback;
}

}  // namespace onnxruntime
