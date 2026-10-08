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

// Matching is case-sensitive. Empty or unrecognized values preserve Auto.
inline KernelDispatchPolicy ParseKernelDispatchPolicy(std::string_view value) {
  if (value == "latency") return KernelDispatchPolicy::Latency;
  if (value == "memory") return KernelDispatchPolicy::Memory;
  if (value == "safe") return KernelDispatchPolicy::Safe;
  return KernelDispatchPolicy::Auto;
}

}  // namespace onnxruntime
