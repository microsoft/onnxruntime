// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/get_env_var.h"

#include <iostream>

#include <Windows.h>

namespace onnxruntime {
optional<std::string> GetEnvironmentVar(const std::string& var_name) {
  // Why getenv() should be avoided on Windows:
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/getenv-wgetenv
  // Instead use the Win32 API: GetEnvironmentVariableA()

  // Max limit of an environment variable on Windows including the null-terminating character
  constexpr DWORD kBufferSize = 32767;

  // Create buffer to hold the result
  char buffer[kBufferSize];

  const auto char_count = ::GetEnvironmentVariableA(var_name.c_str(), buffer, kBufferSize);

  // Will be > 0 if the API call was successful
  if (char_count) {
    return std::string(buffer, buffer + char_count);
  }

  if (const auto err = ::GetLastError(); err != ERROR_ENVVAR_NOT_FOUND) {
    std::cerr << __FUNCTION__ ": GetEnvironmentVariableA() failed with error code: " << err << "\n";
  }

  return nullopt;
}
}  // namespace onnxruntime
