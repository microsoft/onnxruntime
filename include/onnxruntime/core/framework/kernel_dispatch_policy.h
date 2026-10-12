// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

namespace onnxruntime {

// Hardware-neutral intent resolved by each participating operator/EP.
enum class KernelDispatchPolicy {
  Auto,
  Latency,
  Memory,
  Safe,
};

// Matching is case-sensitive. Empty or unrecognized values preserve Auto.
inline KernelDispatchPolicy ParseKernelDispatchPolicy(std::string_view value) {
  if (value == "latency") return KernelDispatchPolicy::Latency;
  if (value == "memory") return KernelDispatchPolicy::Memory;
  if (value == "safe") return KernelDispatchPolicy::Safe;
  return KernelDispatchPolicy::Auto;
}

constexpr const char* KernelDispatchPolicyName(KernelDispatchPolicy policy) {
  switch (policy) {
    case KernelDispatchPolicy::Latency:
      return "latency";
    case KernelDispatchPolicy::Memory:
      return "memory";
    case KernelDispatchPolicy::Safe:
      return "safe";
    case KernelDispatchPolicy::Auto:
    default:
      return "auto";
  }
}

}  // namespace onnxruntime
