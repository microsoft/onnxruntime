// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Cross-platform helper for reading environment variables.
// Shared by platform/windows/env.cc (WindowsEnv::GetEnvironmentVar) and
// core/providers/cuda/plugin/provider_api_shims.cc to avoid code duplication.

#pragma once

#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <cstdlib>
#endif

namespace onnxruntime {
namespace detail {

inline std::string GetEnvironmentVar(const std::string& var_name) {
#ifdef _WIN32
  // Why getenv() should be avoided on Windows:
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/getenv-wgetenv
  // Instead use the Win32 API: GetEnvironmentVariableA()

  // Max limit of an environment variable on Windows including the null-terminating character.
  constexpr DWORD kBufferSize = 32767;

  // Create buffer to hold the result.
  std::string buffer(kBufferSize, '\0');

  // On success, returns the number of characters stored (not including the null terminator).
  // With kBufferSize set to the Windows maximum of 32767, the buffer is always large enough
  // so char_count will always be less than kBufferSize. The comparison is retained for
  // correctness checking and to avoid compiler warnings about an ignored return value.
  auto char_count = GetEnvironmentVariableA(var_name.c_str(), buffer.data(), kBufferSize);

  if (kBufferSize > char_count) {
    buffer.resize(char_count);
    return buffer;
  }

  // Either the call failed (e.g. ERROR_ENVVAR_NOT_FOUND) or the buffer was not large enough.
  return std::string();
#else
  const char* val = std::getenv(var_name.c_str());
  return val == nullptr ? std::string() : std::string(val);
#endif
}

}  // namespace detail
}  // namespace onnxruntime
