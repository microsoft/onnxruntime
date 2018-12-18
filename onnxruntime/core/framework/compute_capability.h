// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {
// A structure encodes a subgraph and the method to run it.
struct ComputeCapability {
  // The subgraph that an XP can execute, it could contain a single node
  // or multiple nodes.
  std::unique_ptr<IndexedSubGraph> sub_graph;

  // When an execution provider fuses a subgraph into a kernel, it passes
  // a kernel create function to onnxruntime so the runtime can create the
  // compute kernel for the subgraph. Otherwise onnxruntime will search
  // kernels in pre-defined kernel registry provided by XP.
  KernelCreateFn fuse_kernel_function;

  // TODO: if there is a FusedKernelFn attached, onnxruntime will generate
  // the default KernelDefinition for it, according to the OpSchema it
  // auto-generates. An execution provider can further set some advanced
  // fields on kernel definition, such as  memory placement / in-place
  // annotation.  
  ComputeCapability() : sub_graph(nullptr), fuse_kernel_function(nullptr) {}

  ComputeCapability(std::unique_ptr<IndexedSubGraph> t_sub_graph,
                    KernelCreateFn t_kernel_func)
      : sub_graph(std::move(t_sub_graph)),
        fuse_kernel_function(t_kernel_func) {}
};
}  // namespace onnxruntime
