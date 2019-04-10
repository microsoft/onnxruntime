// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/function.h"

namespace onnxruntime {
class Graph;
class Node;
class Model;
}  // namespace onnxruntime

namespace onnxruntime {

// Function representation class.
class FunctionImpl final : public Function {
 public:
  FunctionImpl(const onnxruntime::Graph& graph,
               std::unique_ptr<IndexedSubGraph> customized_func);

  FunctionImpl(const onnxruntime::Graph& graph,
               const onnxruntime::NodeIndex& node_index,
               const ONNX_NAMESPACE::FunctionProto* onnx_func);

  ~FunctionImpl() override;

  const ONNX_NAMESPACE::OpSchema& OpSchema() const override;

  const onnxruntime::Graph& Body() const override;

  const IndexedSubGraph& GetIndexedSubGraph() const override;

  const ONNX_NAMESPACE::FunctionProto* GetFuncProto() const;

 private:
  const onnxruntime::Graph* const parent_graph_;
  std::unique_ptr<IndexedSubGraph> customized_func_body_;
  std::unique_ptr<ONNX_NAMESPACE::OpSchema> op_schema_;
  std::unique_ptr<onnxruntime::Model> body_;
  const ONNX_NAMESPACE::FunctionProto* onnx_func_proto_;
};

}  // namespace onnxruntime
