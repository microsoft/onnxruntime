// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <optional>

namespace onnxruntime {

// Phase-A memory roadmap (issue microsoft/onnxruntime#29775). Describes one transient device scratch
// ("workspace") buffer that a kernel needs during Compute() in addition to its output tensor(s) -
// e.g. a split-K / stream-K partial-sum reduction buffer for a GEMM. A workspace is allocated, used
// during a single Compute() call, and freed; it is NOT the output and NOT the weights.
//
// This POD lives in its own lightweight standalone header (pulling in no heavy framework headers) so
// that BOTH OpKernel hierarchies can include it without violating the adapter boundary:
//   - the in-tree onnxruntime::OpKernel (core/framework/op_kernel.h), and
//   - the plugin adapter onnxruntime::ep::adapter::OpKernel (ep/adapter/op_kernel.h), which
//     deliberately does not include core/framework/op_kernel.h.
struct WorkspaceRequirement {
  size_t size_bytes;  // upper-bound scratch bytes for this slot
  int slot_id;        // kernel-defined, stable across runs; unique within one kernel instance

  // Optional pointer-alignment requirement for this slot's buffer, in bytes (e.g. 128, 256). Unset
  // (nullopt) means "the allocator's default alignment is sufficient" - true for every kernel using
  // this API today, since CUDA's allocator already guarantees >= 256-byte aligned allocations and no
  // current workspace consumer needs a stricter or non-default alignment. This field exists so a
  // future kernel with an exotic alignment need (e.g. a stricter tensor-core/WMMA layout constraint),
  // or a future shared-arena packer that co-locates multiple kernels' slots into one allocation (see
  // the "shared per-node memory-budget tracker" direction in issue #29775), can express that need
  // without an ABI-breaking change to this struct. No kernel currently sets this.
  std::optional<size_t> alignment;
};

}  // namespace onnxruntime
