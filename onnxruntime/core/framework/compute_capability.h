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

  // When an execution provider fuses a subgraph into a kernel,
  // there are two ways to run the subgraph: with a predefined kernel,
  // or with code-generation that compile the subgraph at runtime.
  // if this flag is true, execution provider's Compile method will be invoked
  // if onnxruntime decide to assign the subgraph to execution provider.
  bool need_compile;

  // TODO: if there is a FusedKernelFn attached, onnxruntime will generate
  // the default KernelDefinition for it, according to the OpSchema it
  // auto-generates. An execution provider can further set some advanced
  // fields on kernel definition, such as  memory placement / in-place
  // annotation.
  ComputeCapability() : sub_graph(nullptr), need_compile(false) {}

  ComputeCapability(std::unique_ptr<IndexedSubGraph> t_sub_graph,
                    bool compile_flag)
      : sub_graph(std::move(t_sub_graph)),
        need_compile(compile_flag) {}
};
}  // namespace onnxruntime
