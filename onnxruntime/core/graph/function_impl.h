// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/logging/logging.h"
#include "core/graph/function.h"
#include "core/graph/graph.h"

namespace onnxruntime {
class Graph;
class Node;
}  // namespace onnxruntime

namespace onnxruntime {

// Function representation class.
class FunctionImpl final : public Function {
 public:
  // This constructor is used during subgraph fusion in
  // graph partitioning phase. This constructor takes the nodes
  // which need to be fused and creates a function body for the fused node.
  FunctionImpl(onnxruntime::Graph& graph,
               const IndexedSubGraph& nodes_to_fuse);

  // This constructor is used during function body initialization for
  // a Function Op. This takes in a FunctionProto and constructs function body
  // from it. The function body initialization happens during model load in graph resolve
  // phase.
  // model_local_functions contains domain:optype to model_local_functions map. This is
  // used to resolve and initialize nested functions.
  FunctionImpl(onnxruntime::Graph& graph,
               const ONNX_NAMESPACE::FunctionProto& onnx_func);


  ~FunctionImpl() override;

  const onnxruntime::Graph& Body() const override;

  onnxruntime::Graph& MutableBody() override;

 private:
  //onnxruntime::Model body_;
  ONNX_NAMESPACE::GraphProto function_storage_proto_;
  onnxruntime::Graph function_body_graph_;
};

namespace function_utils {

}

}  // namespace onnxruntime
