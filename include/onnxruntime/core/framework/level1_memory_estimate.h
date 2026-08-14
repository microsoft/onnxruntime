// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <optional>

namespace onnxruntime {

// Partition-time memory estimate for allocations whose sizes are known before
// kernel creation. The fields remain separate because they have different
// lifetimes even while the current byte-count accountant conservatively charges
// all of them to the node's budget.
struct Level1MemoryEstimate {
  // Temporary workspace used while executing the kernel. nullopt means that
  // runtime workspace is not estimable and the accountant must use its fallback.
  std::optional<size_t> runtime_workspace_bytes;

  // Kernel-owned prepacked buffers that remain live for the session.
  size_t persistent_prepack_bytes = 0;

  // Scratch buffers used only while constructing persistent prepacked buffers.
  size_t temporary_prepack_bytes = 0;
};

}  // namespace onnxruntime
