// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <optional>

namespace onnxruntime {

// Partition-time memory estimate for allocations whose sizes are known before
// kernel creation. The fields remain separate because they have different
// lifetimes even while the current byte-count accountant conservatively charges
// all of them to the node's budget. Prepack memory contributes to a conservative
// initialization-time upper bound, not exact lifetime-aware or steady-state
// accounting. The accountant separately charges original initializers at their
// planned device locations before PrePack() runs. These fields must therefore
// describe only additional destination and scratch allocations that can coexist
// with those initializers; a destination reused directly from an initializer
// (for example, an offline-prepacked weight) must not be reported again. The
// accountant does not subtract a source initializer after its final prepack
// consumer releases it.
struct Level1MemoryEstimate {
  // Temporary workspace used while executing the kernel. nullopt means that
  // runtime workspace is not estimable and the accountant must use its fallback.
  std::optional<size_t> runtime_workspace_bytes;

  // Kernel-owned prepacked buffers that remain live for the session.
  size_t persistent_prepack_bytes = 0;

  // Initialization-only scratch, including prepack conversion and constructor-time profiling.
  size_t temporary_prepack_bytes = 0;
};

}  // namespace onnxruntime
