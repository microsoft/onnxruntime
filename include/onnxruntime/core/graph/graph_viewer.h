// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"

namespace onnxruntime {
class Function;
struct IndexedSubGraph;
}  // namespace onnxruntime

namespace onnxruntime {

// use value-based compare to make sure transformer output order is consistent
struct NodeCompare {
  bool operator()(const Node* n1, const Node* n2) const;
};

/**
@class GraphViewer
Class that provides a read-only view of the Graph.
@remarks If the underlying Graph is changed, GetNodesInTopologicalOrder and GetRootNodes may become invalid.
*/
class GraphViewer {
 public:
  /**
  Construct a GraphViewer from the provided Graph instance.
  */
  explicit GraphViewer(const Graph& graph);

  /** Gets the Graph name. */
  const std::string& Name() const noexcept;

  /** Gets the Graph description. */
  const std::string& Description() const noexcept;

  /**
  Gets a tensor created from an initializer.
  @param tensor_name The tensor name
  @param[out] value Sets the pointer to the TensorProto if found, or nullptr if not.
  @returns True if found. False if not.
  */
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const;

  /** Returns true if an initializer value can be overridden by a graph input with the same name. */
  bool CanOverrideInitializer() const noexcept;

  /**
  Gets the Graph inputs, excluding initializers.
  @returns Collection of NodeArg pointers for the graph inputs, excluding inputs that have matching initializers.
  @remarks No nullptr values in the returned collection. The order will be the same as in the GraphProto.
  */
  const std::vector<const NodeArg*>& GetInputs() const noexcept;

  /**
  Gets the Graph inputs, including any initializers.
  @returns Collection of NodeArg pointers for all the graph inputs.
  @remarks No nullptr values in the returned collection. The order will be the same as in the GraphProto.
  */
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept;

  /**
  Gets the Graph outputs.
  @returns Collection of NodeArg pointers for all the graph outputs.
  @remarks No nullptr values in the returned collection. The order will be the same as in the GraphProto.
  */
  const std::vector<const NodeArg*>& GetOutputs() const noexcept;

  /** Gets all ValueInfo NodeArg instances in the Graph. */
  const std::vector<const NodeArg*>& GetValueInfo() const noexcept;

  /**
  Gets the Node instance at the specified index.
  @param node_index Index to retrieve Node from.
  @remarks May return nullptr if index no longer points to a valid node due to the node being freed.
  */
  const Node* GetNode(NodeIndex node_index) const;

  /**  Gets an iterator over all the valid Nodes in the Graph. */
  const GraphNodes& Nodes() const noexcept;

  /** Gets the number of valid nodes in the Graph. */
  int NumberOfNodes() const noexcept;

  /** Gets the maximum NodeIndex value used by Nodes in the Graph. */
  int MaxNodeIndex() const noexcept;

  /** Gets the NodeIndex values for the Graph nodes, sorted into topological order. */
  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const;

  /**
  Gets the NodeIndex values for the root nodes in the Graph.
  The root nodes are the topmost nodes in the Graph that receive inputs from the Graph inputs
  and no other nodes in the Graph.
  */
  const std::vector<NodeIndex>& GetRootNodes() const;

  /** Gets all tensors created from initializers. */
  const InitializedTensorSet& GetAllInitializedTensors() const noexcept;

  /**
  Gets the NodeArg instance for the given name.
  @returns A NodeArg if found, a nullptr if not.
  */
  const NodeArg* GetNodeArg(const std::string& name) const;

  /** Gets the map of operator domains to their opset versions. */
  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept {
    return graph_->DomainToVersionMap();
  }

  /** Checks if this is a Subgraph */
  bool IsSubgraph() const;

  /** Get the internal graph*/
  const Graph& GetGraph() const { return *graph_; }

  /**
  returns true if 'name' is an initializer, and is constant and cannot be overridden at runtime.
  @param check_outer_scope If true and the 'graph_' is a subgraph, check parent graph/s for 'name' if not found in 'graph_'.
  */
  bool IsConstantInitializer(const std::string& name, bool check_outer_scope) const;

  /** Get the Node containing this Graph if IsSubgraph is true. Returns nullptr otherwise. */
  const Node* ParentNode() const noexcept { return graph_->ParentNode(); }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphViewer);

  const Graph* graph_;

  // The NodeIndex values of the graph nodes sorted in topological order.
  std::vector<NodeIndex> nodes_in_topological_order_;
  // Graph root nodes.
  std::vector<NodeIndex> root_nodes_;
};
}  // namespace onnxruntime
