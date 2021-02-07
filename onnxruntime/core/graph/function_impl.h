// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/logging/logging.h"
#include "core/graph/function.h"
#include "core/graph/model.h"

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
  FunctionImpl(const onnxruntime::Graph& graph,
               const IndexedSubGraph& nodes_to_fuse,
               const logging::Logger& logger);

  // This constructor is used during function body initialization for
  // a Function Op. This takes in a FunctionProto and constructs function body
  // from it. The function body initialization happens during model load in graph resolve
  // phase.
  FunctionImpl(const onnxruntime::Graph& graph,
               const onnxruntime::NodeIndex& node_index,
               const ONNX_NAMESPACE::FunctionProto& onnx_func,
               const logging::Logger& logger);

  ~FunctionImpl() override;

  const ONNX_NAMESPACE::OpSchema& OpSchema() const override;

  const onnxruntime::Graph& Body() const override;

 private:
  const onnxruntime::Graph* const parent_graph_;
  std::unique_ptr<ONNX_NAMESPACE::OpSchema> op_schema_;
  onnxruntime::Model body_;
  ONNX_NAMESPACE::FunctionProto onnx_func_proto_;
};

// Function that uses a GraphViewer so does not need to build a new Model. We still need the OpSchema to be available
// though so we just create that.
class ViewerFunctionImpl final : public Function {
 public:
  ViewerFunctionImpl(const onnxruntime::Graph& graph,
                     const IndexedSubGraph& nodes_to_fuse,
                     const logging::Logger& logger);

  ~ViewerFunctionImpl() override;

  const ONNX_NAMESPACE::OpSchema& OpSchema() const override { return *op_schema_; }

  const onnxruntime::Graph& Body() const override { ORT_THROW("Not supported"); }

 private:
  std::unique_ptr<ONNX_NAMESPACE::OpSchema> op_schema_;
};

}  // namespace onnxruntime
