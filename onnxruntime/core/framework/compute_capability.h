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

  // TODO: if there is a FusedKernelFn attached, onnxruntime will generate
  // the default KernelDefinition for it, according to the OpSchema it
  // auto-generates. An execution provider can further set some advanced
  // fields on kernel definition, such as  memory placement / in-place
  // annotation.
  ComputeCapability() : sub_graph(nullptr){}

  ComputeCapability(std::unique_ptr<IndexedSubGraph> t_sub_graph)
      : sub_graph(std::move(t_sub_graph)){}
};
}  // namespace onnxruntime
