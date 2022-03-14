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
  // model_local_functions contains domain:optype to model_local_functions map. This is
  // used to resolve and initialize nested functions.
  FunctionImpl(onnxruntime::Graph& graph,
               const onnxruntime::NodeIndex& node_index,
               const ONNX_NAMESPACE::FunctionProto& onnx_func,
               const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& in_model_function_protos,
               std::vector<std::unique_ptr<onnxruntime::Function>>& function_container,
               const logging::Logger& logger,
               bool is_nested_function = false);

  ~FunctionImpl() override;

  const ONNX_NAMESPACE::OpSchema& OpSchema() const override;

  const onnxruntime::Graph& Body() const override;

  onnxruntime::Graph& MutableBody() override;

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

  onnxruntime::Graph& MutableBody() override { ORT_THROW("Not supported"); }

 private:
  std::unique_ptr<ONNX_NAMESPACE::OpSchema> op_schema_;
};

namespace function_utils {
/** Get the unique id for function. This is used as a key to find the 
* relevant model local function from it's container.
* @param function_domain Domain for the function.
* @param function_name Name of the function. Name should match the OpType of the node which references the function.
*/
inline std::string GetFunctionIdentifier(const std::string& function_domain, const std::string& function_name) {
  return function_domain + ":" + function_name;
}
}

}  // namespace onnxruntime
