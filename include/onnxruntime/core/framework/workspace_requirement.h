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

  // Pointer-alignment requirement for this slot's buffer, in bytes (e.g. 128, 256). 0 requests the
  // allocator's default alignment. Any nonzero value is a binding requirement on the returned
  // buffer/root that a planner must honor.
  // A plain size_t (not std::optional<size_t>) is used deliberately: this struct is meant to be usable
  // across a plugin-DLL boundary eventually, and std::optional's layout is not guaranteed stable across
  // compilers/STL versions the way a scalar with a sentinel value is.
  size_t alignment_bytes;
};

}  // namespace onnxruntime
