// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <string>

#include "core/graph/basic_types.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime::graph_utils {

/**
 * Represents an edge between two graph nodes, a graph input or initializer -> node connection, or a node -> graph
 * output connection.
 * Similar to graph_utils::GraphEdge.
 */
struct ExtendedGraphEdge {
  enum class End { Source,
                   Destination };

  /** Information about a source or destination node of the extended edge. */
  struct NodeInfo {
    /** Node index in the graph. */
    NodeIndex node_idx;
    /** Node input or output def index. */
    int arg_idx;
  };

  /** Source node info. If empty, the source is a graph input or initializer. */
  std::optional<NodeInfo> src;
  /** Destination node info. If empty, the destination is a graph output. */
  std::optional<NodeInfo> dst;
  /** Edge/connection NodeArg name. */
  std::string arg_name;

  /** Whether this extended edge has a graph input or initializer as the source. */
  bool HasGraphInputOrInitializer() const noexcept { return !src.has_value(); }

  /** Whether this extended edge has a graph output as the destination. */
  bool HasGraphOutput() const noexcept { return !dst.has_value(); }

  /** Gets the NodeInfo at the specified extended edge end. */
  const std::optional<NodeInfo>& GetNodeInfoAtEnd(End end) const noexcept {
    return end == End::Source ? src : dst;
  }

  /** Gets the graph node at the specified end if it exists, otherwise nullptr. */
  const Node* GetNodeAtEnd(const Graph& graph, End end) const {
    if (const auto& node_info = GetNodeInfoAtEnd(end); node_info.has_value()) {
      const Node* node = graph.GetNode(node_info->node_idx);
      ORT_ENFORCE(node != nullptr, "Invalid node index ", node_info->node_idx);
      return node;
    }
    return nullptr;
  }

  /** Gets the mutable graph node at the specified end if it exists, otherwise nullptr. */
  Node* GetMutableNodeAtEnd(Graph& graph, End end) const {
    if (const auto& node_info = GetNodeInfoAtEnd(end); node_info.has_value()) {
      Node* node = graph.GetNode(node_info->node_idx);
      ORT_ENFORCE(node != nullptr, "Invalid node index ", node_info->node_idx);
      return node;
    }
    return nullptr;
  }

  /**
   * Creates an extended graph edge from a valid graph_utils::GraphEdge.
   * To be valid, `graph_edge` should represent an actual edge between two nodes in the graph.
   * Validity is assumed.
   */
  static ExtendedGraphEdge CreateFromValidGraphEdge(const graph_utils::GraphEdge& graph_edge) {
    return ExtendedGraphEdge{
        NodeInfo{graph_edge.src_node, graph_edge.src_arg_index},
        NodeInfo{graph_edge.dst_node, graph_edge.dst_arg_index},
        graph_edge.arg_name};
  }

  /**
   * Attempts to create an extended graph edge from a graph input or initializer to a node.
   * Returns nullopt if there is not actually a graph input or initializer at the specified node's input index.
   */
  static std::optional<ExtendedGraphEdge> TryCreateFromInputOrInitializerToNode(
      const Graph& graph, const Node& node, int node_input_def_idx) {
    const auto node_inputs = node.InputDefs();
    ORT_ENFORCE(node_input_def_idx >= 0 &&
                static_cast<size_t>(node_input_def_idx) < node_inputs.size());

    const auto* node_input = node_inputs[node_input_def_idx];
    if (!graph.IsInputsIncludingInitializers(node_input)) {
      return std::nullopt;
    }

    return ExtendedGraphEdge{
        std::nullopt,
        NodeInfo{node.Index(), node_input_def_idx},
        node_input->Name()};
  }

  /**
   * Attempts to create an extended graph edge from a node to a graph output.
   * Returns nullopt if there is not actually a graph output at the specified node's output index.
   */
  static std::optional<ExtendedGraphEdge> TryCreateFromNodeToOutput(
      const Graph& graph, const Node& node, int node_output_def_idx) {
    const auto node_outputs = node.OutputDefs();
    ORT_ENFORCE(node_output_def_idx >= 0 &&
                static_cast<size_t>(node_output_def_idx) < node_outputs.size());

    const auto* node_output = node_outputs[node_output_def_idx];
    if (!graph.IsOutput(node_output)) {
      return std::nullopt;
    }

    return ExtendedGraphEdge{
        NodeInfo{node.Index(), node_output_def_idx},
        std::nullopt,
        node_output->Name()};
  }

  // there is also the case where a graph input or initializer is an output
  // if that's useful to represent, add another creation function
};

}  // namespace onnxruntime::graph_utils
