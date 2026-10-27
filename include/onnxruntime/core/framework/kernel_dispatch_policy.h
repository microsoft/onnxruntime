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

// Matching is case-sensitive. Empty and "auto" select Auto; failure leaves policy unchanged.
inline bool TryParseKernelDispatchPolicy(
    std::string_view value,
    KernelDispatchPolicy& policy) noexcept {
  if (value.empty() || value == "auto") {
    policy = KernelDispatchPolicy::Auto;
    return true;
  }
  if (value == "latency") {
    policy = KernelDispatchPolicy::Latency;
    return true;
  }
  if (value == "memory") {
    policy = KernelDispatchPolicy::Memory;
    return true;
  }
  if (value == "safe") {
    policy = KernelDispatchPolicy::Safe;
    return true;
  }
  return false;
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
