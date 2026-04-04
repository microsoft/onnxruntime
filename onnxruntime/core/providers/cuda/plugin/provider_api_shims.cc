// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Provider API shims used by migrated CUDA kernels.
// Provides direct implementations of utility functions that in-tree kernels
// obtain via the SHARED_PROVIDER bridge (GetEnvironmentVar, floatToHalf,
// halfToFloat). Plugin builds skip SHARED_PROVIDER entirely, so these thin
// wrappers ensure the migrated kernel code compiles and links.

#include <string>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#endif
#include "core/common/float16.h"

namespace onnxruntime {

std::string GetEnvironmentVar(const std::string& var_name) {
#ifdef _WIN32
  // getenv() should be avoided on Windows; use the Win32 API instead.
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/getenv-wgetenv
  constexpr DWORD kBufferSize = 32767;
  std::string buffer(kBufferSize, '\0');
  auto char_count = GetEnvironmentVariableA(var_name.c_str(), buffer.data(), kBufferSize);
  if (kBufferSize > char_count) {
    buffer.resize(char_count);
    return buffer;
  }
  return std::string();
#else
  const char* val = std::getenv(var_name.c_str());
  return val ? std::string(val) : std::string();
#endif
}

namespace math {

uint16_t floatToHalf(float f) {
  return MLFloat16(f).val;
}

float halfToFloat(uint16_t h) {
  return MLFloat16::FromBits(h).ToFloat();
}

}  // namespace math

}  // namespace onnxruntime
