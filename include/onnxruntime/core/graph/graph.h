// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_base.h"

namespace onnxruntime {
class Function;
struct IndexedSubGraph;
}  // namespace onnxruntime

namespace onnxruntime {
struct FunctionContainer;
// A graph viewer representation class.
class GraphViewer {
 public:
  GraphViewer(const Graph& graph);

  // Graph name.
  const std::string& Name() const noexcept;

  const std::string& Description() const noexcept;

  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const;

  // Graph inputs excluding initializers.
  const std::vector<const NodeArg*>& GetInputs() const noexcept;
  // Graph inputs including initializers. Contains no nullptr values.
  // This will match the number and order of inputs from the GraphProto.
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept;

  // Graph outputs. Should have no nullptr values.
  const std::vector<const NodeArg*>& GetOutputs() const noexcept;

  // Get graph value infos.
  const std::vector<const NodeArg*>& GetValueInfo() const noexcept;

  // Get const Node given specific node index. May return nullptr if node as been freed.
  const Node* GetNode(NodeIndex node_index) const;

  const GraphNodes& Nodes() const noexcept;

  int NumberOfNodes() const noexcept;

  int MaxNodeIndex() const noexcept;

  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const;

  const std::vector<NodeIndex>& GetRootNodes() const;

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphViewer);

  const Graph* graph_;

  // The topological order of node index.
  std::vector<NodeIndex> nodes_in_topological_order_;
  // Graph root nodes.
  std::vector<NodeIndex> root_nodes_;
};
}  // namespace onnxruntime
