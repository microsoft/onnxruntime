// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

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

  // Pointer-alignment requirement for this slot's buffer, in bytes (e.g. 128, 256). 0 means "the
  // allocator's default alignment is sufficient" - true for every kernel using this API today, since
  // CUDA's allocator already guarantees >= 256-byte aligned allocations. Reserved for a future kernel
  // with a stricter/exotic alignment need, or a future shared-arena packer that co-locates multiple
  // kernels' slots (see the "shared per-node memory-budget tracker" direction in issue #29775).
  // A plain size_t (not std::optional<size_t>) is used deliberately: this struct is meant to be usable
  // across a plugin-DLL boundary eventually, and std::optional's layout is not guaranteed stable across
  // compilers/STL versions the way a scalar with a sentinel value is.
  size_t alignment_bytes;
};

}  // namespace onnxruntime
