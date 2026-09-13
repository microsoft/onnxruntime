// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <optional>

namespace onnxruntime {

// Partition-time memory estimate for allocations whose sizes are known before
// kernel creation. The fields remain separate because they have different
// lifetimes. Persistent memory is charged to the additive node budget, while
// initialization scratch is reported as a session-wide peak because kernel
// construction and PrePack() are sequential. The accountant separately charges
// original initializers at their planned device locations before PrePack() runs.
// These fields must therefore
// describe only additional destination and scratch allocations that can coexist
// with those initializers; a destination reused directly from an initializer
// (for example, an offline-prepacked weight) must not be reported again. The
// accountant does not subtract a source initializer after its final prepack
// consumer releases it.
struct Level1MemoryEstimate {
  // Temporary workspace used by normal kernel execution. nullopt means that
  // runtime workspace is not estimable and the accountant must use its fallback.
  std::optional<size_t> runtime_workspace_bytes;

  // Kernel-owned prepacked buffers that remain live for the session.
  size_t persistent_prepack_bytes = 0;

  // Initialization-only scratch, including prepack conversion and constructor-time profiling.
  size_t initialization_scratch_bytes = 0;

  // Temporary allocation used by runtime preparation outside normal kernel execution, such as
  // lazy tactic profiling for a previously unseen input-size bucket. Because profiling completes
  // before normal execution, this allocation does not overlap runtime_workspace_bytes. The
  // accountant therefore uses their peak (or the peak with fallback workspace) instead of
  // summing them:
  //   runtime_peak_bytes = std::max(runtime_workspace_bytes, runtime_transient_bytes)
  size_t runtime_transient_bytes = 0;
};

}  // namespace onnxruntime
