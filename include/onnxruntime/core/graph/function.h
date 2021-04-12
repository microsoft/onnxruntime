// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {
class Graph;
class Node;
}  // namespace onnxruntime

namespace onnxruntime {

/** 
@class Function 
Class representing a Function.
*/
class Function {
 public:
  virtual ~Function() = default;

#if !defined(ORT_MINIMAL_BUILD)
  /** Gets the OpSchema for the Function. */
  virtual const ONNX_NAMESPACE::OpSchema& OpSchema() const = 0;
#endif

  /** Gets the Graph instance for the Function body subgraph. */
  virtual const onnxruntime::Graph& Body() const = 0;
};

/** 
Create a new Function instance.
@param graph The graph containing the Function.
@param nodes_to_fuse the IndexedSubGraph to use for the Function.
*/
std::unique_ptr<Function> MakeFunction(const onnxruntime::Graph& graph,
                                       const IndexedSubGraph& nodes_to_fuse,
                                       const logging::Logger& logger);
}  // namespace onnxruntime
