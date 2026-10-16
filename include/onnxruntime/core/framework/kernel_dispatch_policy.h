// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

namespace onnxruntime {

// Hardware-neutral intent for future operator/EP policy resolution.
// Reader-only: no value currently changes kernel dispatch or workspace sizing.
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

}  // namespace onnxruntime
