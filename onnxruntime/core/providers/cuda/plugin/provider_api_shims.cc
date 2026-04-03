// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Provider API shims used by migrated CUDA kernels.
// Provides direct implementations of utility functions that in-tree kernels
// obtain via the SHARED_PROVIDER bridge (GetEnvironmentVar, floatToHalf,
// halfToFloat). Plugin builds skip SHARED_PROVIDER entirely, so these thin
// wrappers ensure the migrated kernel code compiles and links.

#include <string>
#include <cstdlib>
#include "core/common/float16.h"

namespace onnxruntime {

std::string GetEnvironmentVar(const std::string& var_name) {
#ifdef _MSC_VER
  char* buf = nullptr;
  size_t len = 0;
  _dupenv_s(&buf, &len, var_name.c_str());
  std::string result = buf ? std::string(buf) : std::string();
  free(buf);
  return result;
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
