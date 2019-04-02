// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

namespace graph_utils {

// local helpers

// check if an output edge provides an implicit input to the destination node
static bool OutputEdgeProvidesImplicitInput(const Graph& graph, const GraphEdge& output_edge) {
  // we treat the explicit and implicit inputs as sequential, so if the destination arg index of an output edge
  // is past the valid range for the node's explicit inputs, it is for an implicit input
  const auto num_explicit_inputs = (*graph.GetNode(output_edge.dst_node)).InputDefs().size();
  bool is_implicit_input = output_edge.dst_arg_index >= num_explicit_inputs;
  return is_implicit_input;
}

// Get the name of the outgoing NodeArg with the specified index for the given node.
static const std::string& GetNodeOutputName(const Node& node, int index) {
  const auto& outputs = node.OutputDefs();

  // this should never happen as it's internal logic so just use an assert
  assert(index < outputs.size());

  return outputs[index]->Name();
}

// fusion is only done for ONNX domain ops
bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       ONNX_NAMESPACE::OperatorSetVersion version,
                                       const std::string& domain) {
  if (node.OpType() != op_type ||
      node.Op()->Deprecated() || node.Op()->SinceVersion() != version ||
      (!node.Domain().empty() && node.Domain() != domain)) {
    return false;
  }
  return true;
}

bool IsSupportedProvider(const Node& node,
                         const std::unordered_set<std::string>& compatible_providers) {
  if (!compatible_providers.empty() &&
      compatible_providers.find(node.GetExecutionProviderType()) == compatible_providers.end()) {
    return false;
  }

  return true;
}

Status ForAllMutableSubgraphs(Graph& graph, std::function<Status(Graph&)> func) {
  Status status = Status::OK();

  for (auto& node : graph.Nodes()) {
    for (auto& attr_name_to_subgraph_pair : node.GetAttributeNameToMutableSubgraphMap()) {
      Graph* subgraph = attr_name_to_subgraph_pair.second;
      ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

      status = func(*subgraph);
      ORT_RETURN_IF_ERROR(status);

      // recurse
      status = ForAllMutableSubgraphs(*subgraph, func);
      ORT_RETURN_IF_ERROR(status);
    }
  }

  return status;
}

Status ForAllSubgraphs(const Graph& graph, std::function<Status(const Graph&)> func) {
  Status status = Status::OK();

  for (auto& node : graph.Nodes()) {
    for (auto& attribute : node.GetAttributes()) {
      auto& name = attribute.first;
      auto& proto = attribute.second;

      // check if it has a subgraph
      if (proto.has_g()) {
        const Graph* subgraph = node.GetGraphAttribute(name);
        ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

        status = func(*subgraph);
        ORT_RETURN_IF_ERROR(status);

        // recurse
        status = ForAllSubgraphs(*subgraph, func);
        ORT_RETURN_IF_ERROR(status);
      }
    }
  }

  return status;
}

bool IsSingleInSingleOutNode(const Node& node) {
  return node.GetInputEdgesCount() == 1 && node.GetOutputEdgesCount() == 1;
}

const ONNX_NAMESPACE::AttributeProto* GetNodeAttribute(const Node& node, const std::string& attr_name) {
  const auto& attrs = node.GetAttributes();
  const auto iter = attrs.find(attr_name);
  return iter == attrs.end() ? nullptr : &iter->second;
}

/** Checks if new_output_name can be used to replace removed_output_name in the subgraph input.
    If there is an existing NodeArg in a subgraph that implicitly consumes removed_output_name, it is not safe. */
static bool CanUpdateImplicitInputNameInSubgraph(Node& node,
                                                 const std::string& removed_output_name,
                                                 const std::string& new_output_name) {
  for (auto& attr_subgraph_pair : node.GetAttributeNameToMutableSubgraphMap()) {
    Graph& subgraph = *attr_subgraph_pair.second;
    // if we have an existing NodeArg in the subgraph with the new_output_name that would override an implicit input
    // with the same name
    if (subgraph.GetNodeArg(new_output_name) != nullptr) {
      return false;
    }

    for (auto& subgraph_node : attr_subgraph_pair.second->Nodes()) {
      // recurse if this node also consumes removed_output_name as an implicit input (i.e. there are multiple levels of nested
      // subgraphs, and at least one level lower uses removed_output_name as an implicit input
      const auto& subgraph_node_implicit_inputs = subgraph_node.ImplicitInputDefs();
      if (!subgraph_node_implicit_inputs.empty()) {
        auto subgraph_node_also_consumes_nodearg_as_implicit_input =
            std::find_if(subgraph_node_implicit_inputs.cbegin(), subgraph_node_implicit_inputs.cend(),
                         [&removed_output_name](const NodeArg* input) {
                           return input != nullptr && input->Name() == removed_output_name;
                         });

        if (subgraph_node_also_consumes_nodearg_as_implicit_input != subgraph_node_implicit_inputs.cend()) {
          if (!CanUpdateImplicitInputNameInSubgraph(subgraph_node, removed_output_name, new_output_name))
            return false;
        }
      }
    }
  }

  return true;
}

/** Updates removed_output_name with new_output_name in the subgraph input. */
static void UpdateImplicitInputNameInSubgraph(Node& node,
                                              const std::string& removed_output_name,
                                              const std::string& new_output_name) {
  for (auto& attr_subgraph_pair : node.GetAttributeNameToMutableSubgraphMap()) {
    Graph& subgraph = *attr_subgraph_pair.second;

    for (auto& subgraph_node : subgraph.Nodes()) {
      // recurse if this node also consumes removed_output_name as an implicit input
      // (i.e. there are multiple levels of nested subgraphs, and at least one level lower uses
      // removed_output_name as an implicit input
      const auto& subgraph_node_implicit_inputs = subgraph_node.ImplicitInputDefs();
      if (!subgraph_node_implicit_inputs.empty()) {
        auto subgraph_node_also_consumes_nodearg_as_implicit_input =
            std::find_if(subgraph_node_implicit_inputs.cbegin(), subgraph_node_implicit_inputs.cend(),
                         [&removed_output_name](const NodeArg* input) {
                           return input->Name() == removed_output_name;
                         });

        if (subgraph_node_also_consumes_nodearg_as_implicit_input != subgraph_node_implicit_inputs.cend()) {
          UpdateImplicitInputNameInSubgraph(subgraph_node, removed_output_name, new_output_name);
        }
      }

      // Need mutable input defs to be able to update the implicit input names
      auto& input_args = subgraph_node.MutableInputDefs();

      if (!input_args.empty()) {
        int input_slot_index = -1;
        for (const auto* input_arg : input_args) {
          ++input_slot_index;
          // if the input matches, replace the NodeArg with one using the new name
          if (input_arg->Exists() && input_arg->Name() == removed_output_name) {
            // sanity check there was no edge for this input. implicit inputs from outer scope do not have edges
            ORT_ENFORCE(std::count_if(subgraph_node.InputEdgesBegin(), subgraph_node.InputEdgesEnd(),
                                      [input_slot_index](const Node::EdgeEnd& entry) {
                                        return entry.GetDstArgIndex() == input_slot_index;
                                      }) == 0);

            // Create a new NodeArg with the new name
            input_args[input_slot_index] = &attr_subgraph_pair.second->GetOrCreateNodeArg(new_output_name,
                                                                                          input_arg->TypeAsProto());
          }
        }
      }
    }
  }
}

/** Returns a vector of the output GraphEdges of a node. */
static std::vector<GraphEdge> GetNodeOutputEdges(const Node& node) {
  std::vector<GraphEdge> output_edges;
  for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
    output_edges.push_back(GraphEdge{node, *it, false});
  }

  return output_edges;
}

/** Removes a set of GraphEdges from the graph. */
static void RemoveOutputEdges(Graph& graph, const std::vector<GraphEdge>& edges) {
  for (const auto& edge_to_remove : edges) {
    graph.RemoveEdge(edge_to_remove.src_node,
                     edge_to_remove.dst_node,
                     edge_to_remove.src_arg_index,
                     edge_to_remove.dst_arg_index);
  }
}

/** Removes a node with a single incoming node. */
static bool RemoveNodeWithSingleNodeIn(Graph& graph, Node& node) {
  // Store info for input and output edges.
  std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);
  const Node::EdgeEnd& input_edge_end = *node.InputEdgesBegin();
  const auto& input_edge_node = input_edge_end.GetNode();
  const GraphEdge input_edge{input_edge_node.Index(),
                             node.Index(),
                             input_edge_end.GetSrcArgIndex(),
                             input_edge_end.GetDstArgIndex(),
                             GetNodeOutputName(input_edge_node, input_edge_end.GetSrcArgIndex())};

  // Check if one of the edges provides an implicit input to a subgraph, and if so, check if the subgraph allows
  // the node to be safely removed.
  for (const auto& output_edge : output_edges) {
    if (OutputEdgeProvidesImplicitInput(graph, output_edge)) {
      Node& mutable_output_edge_node = *graph.GetNode(output_edge.dst_node);
      if (!CanUpdateImplicitInputNameInSubgraph(mutable_output_edge_node, output_edge.arg_name, input_edge.arg_name)) {
        LOGS_DEFAULT(WARNING) << " Removal of node " << node.Name()
                              << " failed due to problem in updating a subgraph.";
        return false;
      }
    }
  }

  // Remove the output edges of the node and then the node itself (this will remove its input edge too).
  RemoveOutputEdges(graph, output_edges);
  graph.RemoveNode(node.Index());

  // Create connections between the incoming node and the outgoing nodes of the node that we removed.
  for (const auto& output_edge : output_edges) {
    // Take care of subgraph inputs.
    if (OutputEdgeProvidesImplicitInput(graph, output_edge)) {
      Node& mutable_output_edge_node = *graph.GetNode(output_edge.dst_node);
      UpdateImplicitInputNameInSubgraph(mutable_output_edge_node, output_edge.arg_name, input_edge.arg_name);
    }

    // Add new edge connecting the input with the output nodes directly.
    graph.AddEdge(input_edge.src_node, output_edge.dst_node, input_edge.src_arg_index, output_edge.dst_arg_index);
  }

  return true;
}

/** Remove a node with a single incoming initializer (and no other incoming node). */
static bool RemoveNodeWithSingleInitializerIn(Graph& graph, Node& node) {
  // Store info for input initializer and output edges.
  NodeArg* input_def = node.MutableInputDefs()[0];
  std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);

  // Check if one of the edges provides an implicit input to a subgraph, and if so, check if the subgraph allows
  // the node to be safely removed.
  for (const auto& output_edge : output_edges) {
    if (OutputEdgeProvidesImplicitInput(graph, output_edge)) {
      Node& mutable_output_edge_node = *graph.GetNode(output_edge.dst_node);
      if (!CanUpdateImplicitInputNameInSubgraph(mutable_output_edge_node, output_edge.arg_name, input_def->Name())) {
        LOGS_DEFAULT(WARNING) << " Removal of node " << node.Name()
                              << " failed due to problem in updating a subgraph.";
        return false;
      }
    }
  }

  // Remove the output edges of the node and then the node itself (this will remove its input edge too).
  RemoveOutputEdges(graph, output_edges);
  graph.RemoveNode(node.Index());

  // Add the incoming initializer as input to the outgoing nodes of the node that we removed.
  for (auto& output_edge : output_edges) {
    // Take care of subgraph inputs.
    if (OutputEdgeProvidesImplicitInput(graph, output_edge)) {
      Node& mutable_output_edge_node = *graph.GetNode(output_edge.dst_node);
      UpdateImplicitInputNameInSubgraph(mutable_output_edge_node, output_edge.arg_name, input_def->Name());
    }

    // Replace outgoing node's input to use the initializer.
    auto output_node = graph.GetNode(output_edge.dst_node);
    ORT_ENFORCE(output_node, "Outgoing node could not be found.");

    output_node->MutableInputDefs()[output_edge.dst_arg_index] = input_def;
  }

  return true;
}

bool RemoveSingleInputNode(Graph& graph, Node& node) {
  if (node.GetInputEdgesCount() > 1 || node.InputDefs().size() > 1) {
    return false;
  }

  // If it has a single input that is not an initializer (initializers are not connected with edges to nodes).
  if (node.GetInputEdgesCount() == 1) {
    return RemoveNodeWithSingleNodeIn(graph, node);
  } else if (node.InputDefs().size() == 1) {
    return RemoveNodeWithSingleInitializerIn(graph, node);
  }

  return false;
}

bool IsGraphInput(const Graph& graph, const NodeArg* input) {
  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();
  return std::find(graph_inputs.begin(), graph_inputs.end(), input) != graph_inputs.end();
}

bool AllNodeInputsAreConstant(const Graph& graph, const Node& node) {
  if (node.GetInputEdgesCount() > 0) {
    return false;
  }
  const onnx::TensorProto* initializer = nullptr;
  for (const auto* input_def : node.InputDefs()) {
    // Important note: when an initializer appears in the graph's input, this input will not be considered constant,
    // because it can be overriden by the user at runtime. For constant folding to be applied, the initializer should not
    // appear in the graph's inputs (that is the only way to guarantee it will always be constant).
    if (!graph.GetInitializedTensor(input_def->Name(), initializer) || IsGraphInput(graph, input_def)) {
      return false;
    }
  }
  return true;
}

size_t RemoveNodeOutputEdges(Graph& graph, Node& node) {
  std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);

  for (const auto& edge_to_remove : output_edges) {
    graph.RemoveEdge(edge_to_remove.src_node,
                     edge_to_remove.dst_node,
                     edge_to_remove.src_arg_index,
                     edge_to_remove.dst_arg_index);
  }

  return output_edges.size();
}

}  // namespace graph_utils

}  // namespace onnxruntime
