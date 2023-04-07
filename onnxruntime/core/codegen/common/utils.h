// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include <cassert>
#include <memory>
#include <vector>

namespace onnxruntime {

// Holding utility functions that are not tied to TVM and ORT

std::unique_ptr<char[]> GetEnv(const char* var);

// Check if an environment variable is set
bool IsEnvVarDefined(const char* var);

int64_t TotalSize(const std::vector<int64_t>& shape);

void GetStrides(const int64_t* shape, int ndim, std::vector<int64_t>& strides);

struct TargetFeature {
  bool hasAVX;
  bool hasAVX2;
  bool hasAVX512;
};

TargetFeature GetTargetInfo(const codegen::CodeGenSettings& setttings);

// GCD (Greatest Common Divisor)
template <typename T>
T GCD(T a, T b) {
  ORT_ENFORCE(a >= 0);
  ORT_ENFORCE(b >= 0);
  if (a < b) std::swap(a, b);
  if (b == 0) return a;
  while (a % b != 0) {
    a = a % b;
    std::swap(a, b);
  }
  return b;
}

}  // namespace onnxruntime
