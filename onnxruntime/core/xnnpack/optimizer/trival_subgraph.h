// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <onnx/onnx_pb.h>
#include "core/graph/function.h"

namespace onnxruntime {

class TrivalSubgraph : public Function {
 private:
  std::unique_ptr<ONNX_NAMESPACE::GraphProto> p_;
  Graph g_;

 public:
  TrivalSubgraph(Graph& parent_graph, const Node& parent_node, std::unique_ptr<ONNX_NAMESPACE::GraphProto> p)
      : p_(std::move(p)), g_(parent_graph, parent_node, *p_) {

  };

  /** Gets the Graph instance for the Function body subgraph. */
  const onnxruntime::Graph& Body() const override { return g_; }

  /** Gets the Mutable Graph instance for the Function body subgraph. */
  onnxruntime::Graph& MutableBody() override { return g_; }
};
}  // namespace onnxruntime