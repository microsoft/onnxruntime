// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/common/utils.h"
#include "core/common/cpuid_info.h"
#include "core/common/make_unique.h"

#include <stdlib.h>
#include <string.h>

namespace onnxruntime {

std::unique_ptr<char[]> GetEnv(const char* var) {
  char* val = nullptr;
#if _MSC_VER
  size_t len;

  if (_dupenv_s(&val, &len, var)) {
    // Something went wrong, just return nullptr.
    return nullptr;
  }
#else
  val = getenv(var);
#endif  // _MSC_VER

  if (val == nullptr) {
    return nullptr;
  }

  // On windows, we will have to explicitly free val. Instead of returning val
  // to its caller and make distinguish between windows and linux, we return
  // a unique_ptr, and it will be destroyed automatically after the caller
  // completes.
  size_t len_val = strlen(val) + 1;
  auto p = onnxruntime::make_unique<char[]>(len_val);
  // use explicit loop to get ride of VC's warning on unsafe copy
  for (size_t i = 0; i < len_val; ++i) {
    p[i] = val[i];
  }
  return p;
}

bool IsEnvVarDefined(const char* var) {
  auto val = GetEnv(var);
  return val != nullptr;
}

int64_t TotalSize(const std::vector<int64_t>& shape) {
  int64_t total = 1;
  for (auto s : shape) {
    total *= s;
  }
  return total;
}

// Return the strides for the input shape, i.e. the number of
// elements contained by a single element of current dimension.
// For example, for shape[3][4][5][6], strides will be
// [4*5*6, 5*6, 6, 1], i.e. [120, 30, 6, 1]
void GetStrides(const int64_t* shape, int ndim, std::vector<int64_t>& strides) {
  strides.resize(ndim);
  strides[ndim - 1] = 1;
  for (int64_t i = ndim - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
}

// Common utils to get target option
TargetFeature GetTargetInfo(const codegen::CodeGenSettings& settings) {
  TargetFeature feature;

  std::string target_str = "";
  if (settings.HasOption(nuphar::kNupharCodeGenTarget) && settings.HasOption(nuphar::kNupharCachePath)) {
    target_str = settings.GetOptionValue(nuphar::kNupharCodeGenTarget);
  }

  bool isAVX = false;
  bool isAVX2 = false;
  bool isAVX512 = false;
  if (target_str == "avx") {
    isAVX = true;
  } else if (target_str == "avx2") {
    isAVX = true;
    isAVX2 = true;
  } else if (target_str == "avx512") {
    isAVX = true;
    isAVX2 = true;
    isAVX512 = true;
  } else {
    isAVX = CPUIDInfo::GetCPUIDInfo().HasAVX();
    isAVX2 = CPUIDInfo::GetCPUIDInfo().HasAVX2();
    isAVX512 = CPUIDInfo::GetCPUIDInfo().HasAVX512Skylake();
  }

  feature.hasAVX = isAVX;
  feature.hasAVX2 = isAVX2;
  feature.hasAVX512 = isAVX512;

  return feature;
}

}  // namespace onnxruntime
