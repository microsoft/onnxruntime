// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "core/common/common.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/graph.h"
#include "core/optimizer/graph_optimizer_registry.h"

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
  ComputeCapability() : sub_graph(nullptr) {}

  ComputeCapability(std::unique_ptr<IndexedSubGraph> t_sub_graph)
      : sub_graph(std::move(t_sub_graph)) {}

  // Optional function to optimize this ComputeCapability.
  // This will be called by ORT once the ComputeCapability is assigned to the EP.
  std::function<Status(Graph&,
                       const ComputeCapability& /* this_optimization*/,
                       ComputeCapability& /* cc_to_update */,
                       const GraphOptimizerRegistry&)>
      optimization_func;

  // Optional ComputeCapability instances for sets of nodes within this ComputeCapability that should be optimized.
  // when an optimization is applied, ORT will update this ComputeCapability to reflect the changes made.
  // IndexedSubGraph.nodes:
  //  - update based on RemovedNode/AddNode calls
  // IndexedSubGraph.MetaDef (if present):
  //  - inputs and outputs will be unchanged
  //  - constant_initializers MAY change if we constant fold an initializer during optimization
  std::vector<std::unique_ptr<ComputeCapability>> nodes_to_optimize;
};
}  // namespace onnxruntime
