// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

namespace graph_utils {

struct GraphEdge {
  NodeIndex src_node;
  NodeIndex dst_node;
  int src_arg_index;
  int dst_arg_index;
  std::string arg_name;

  GraphEdge(NodeIndex src_node, NodeIndex dst_node,
            int src_arg_index, int dst_arg_index, const std::string& arg_name) : src_node(src_node),
                                                                                 dst_node(dst_node),
                                                                                 src_arg_index(src_arg_index),
                                                                                 dst_arg_index(dst_arg_index),
                                                                                 arg_name(arg_name) {}

  // Constructs a GraphEdge given a node, an edge_end, and a boolean for the edge direction.
  static GraphEdge CreateGraphEdge(const Node& node, const Node::EdgeEnd& edge_end, bool is_input_edge) {
    return is_input_edge
               ? GraphEdge(edge_end.GetNode().Index(),
                           node.Index(),
                           edge_end.GetSrcArgIndex(),
                           edge_end.GetDstArgIndex(),
                           GetNodeInputName(node, edge_end.GetDstArgIndex()))
               : GraphEdge(node.Index(),
                           edge_end.GetNode().Index(),
                           edge_end.GetSrcArgIndex(),
                           edge_end.GetDstArgIndex(),
                           GetNodeOutputName(node, edge_end.GetSrcArgIndex()));
  }
};

//---------------------
//--- local helpers ---
//---------------------

// check if an output edge provides an implicit input to the destination node
static bool OutputEdgeProvidesImplicitInput(const Graph& graph, const GraphEdge& output_edge) {
  // we treat the explicit and implicit inputs as sequential, so if the destination arg index of an output edge
  // is past the valid range for the node's explicit inputs, it is for an implicit input
  const auto num_explicit_inputs = (*graph.GetNode(output_edge.dst_node)).InputDefs().size();
  bool is_implicit_input = output_edge.dst_arg_index >= num_explicit_inputs;
  return is_implicit_input;
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
      const auto subgraph_node_implicit_inputs = subgraph_node.ImplicitInputDefs();
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
      const auto subgraph_node_implicit_inputs = subgraph_node.ImplicitInputDefs();
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
    output_edges.push_back(GraphEdge::CreateGraphEdge(node, *it, false));
  }

  return output_edges;
}

/** Removes a set of GraphEdges from the graph. */
static void RemoveGraphEdges(Graph& graph, const std::vector<GraphEdge>& edges) {
  for (const auto& edge_to_remove : edges) {
    graph.RemoveEdge(edge_to_remove.src_node,
                     edge_to_remove.dst_node,
                     edge_to_remove.src_arg_index,
                     edge_to_remove.dst_arg_index);
  }
}

/** Given a graph, a list of edges, and a NodeArg name, checks if each of the edges provides an implicit input
    to a subgraph. If so, it checks if there is no clash of the given NodeArg name in each of the subgraphs. 
    This is important when removing a node with this NodeArg as input. */
bool CanUpdateImplicitInputNameInSubgraphs(Graph& graph,
                                           const std::vector<GraphEdge>& output_edges,
                                           const std::string& input_arg_name) {
  for (const auto& output_edge : output_edges) {
    if (OutputEdgeProvidesImplicitInput(graph, output_edge)) {
      Node& mutable_output_edge_node = *graph.GetNode(output_edge.dst_node);
      if (!CanUpdateImplicitInputNameInSubgraph(mutable_output_edge_node, output_edge.arg_name, input_arg_name)) {
        LOGS_DEFAULT(WARNING) << " Implicit input name " << input_arg_name
                              << " cannot be safely updated in one of the subgraphs.";
        return false;
      }
    }
  }
  return true;
}

/** Removes a node with a single incoming node. */
static bool RemoveNodeWithSingleNodeIn(Graph& graph, Node& node) {
  // Store info for input and output edges.
  std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);
  const Node::EdgeEnd& input_edge_end = *node.InputEdgesBegin();
  const GraphEdge input_edge = GraphEdge::CreateGraphEdge(node, input_edge_end, true);

  // Check that the incoming NodeArg can be safely used in the presence of subgraphs.
  if (!CanUpdateImplicitInputNameInSubgraphs(graph, output_edges, input_edge.arg_name)) {
    return false;
  }

  // Remove the output edges of the node and then the node itself (this will remove its input edge too).
  RemoveGraphEdges(graph, output_edges);
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
  auto* input_def = node.MutableInputDefs()[0];
  std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);

  // Check that the incoming NodeArg can be safely used in the presence of subgraphs.
  if (!CanUpdateImplicitInputNameInSubgraphs(graph, output_edges, input_def->Name())) {
    return false;
  }

  // Remove the output edges of the node and then the node itself (this will remove its input edge too).
  RemoveGraphEdges(graph, output_edges);
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

    auto dst_arg_idx = output_edge.dst_arg_index;
    if (dst_arg_idx < output_node->InputDefs().size()) {
      output_node->MutableInputDefs()[output_edge.dst_arg_index] = input_def;
    } else if (dst_arg_idx < output_node->InputDefs().size() + output_node->ImplicitInputDefs().size()) {
      // If we need to update an implicit input.
      output_node->MutableImplicitInputDefs()[dst_arg_idx - output_node->InputDefs().size()] = input_def;
    } else {
      LOGS_DEFAULT(ERROR) << " Invalid value for input index of node " << output_node->Name();
      return false;
    }
  }

  return true;
}

//----------------------------
//--- end of local helpers ---
//----------------------------

const std::string& GetNodeInputName(const Node& node, int index) {
  const auto& inputs = node.InputDefs();
  ORT_ENFORCE(index < inputs.size(), "Attempting to get an input that does not exist.");
  return inputs[index]->Name();
}

const std::string& GetNodeOutputName(const Node& node, int index) {
  const auto& outputs = node.OutputDefs();
  assert(index < outputs.size());
  return outputs[index]->Name();
}

bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       ONNX_NAMESPACE::OperatorSetVersion version,
                                       const std::string& domain) {
  return (node.OpType() == op_type && !node.Op()->Deprecated() &&
          MatchesOpSinceVersion(node, version) && MatchesOpSetDomain(node, domain));
}

bool MatchesOpSinceVersion(const Node& node, ONNX_NAMESPACE::OperatorSetVersion version) {
  return node.Op()->SinceVersion() == version;
}

bool MatchesOpSetDomain(const Node& node, const std::string& domain) {
  const auto& node_domain = node.Domain();
  // We do a special check for the ONNX domain, as it has two aliases.
  return node_domain == domain ||
         ((node_domain == kOnnxDomain || node_domain == kOnnxDomainAlias) &&
          (domain == kOnnxDomain || domain == kOnnxDomainAlias));
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
  return node.InputDefs().size() == 1 &&
         node.ImplicitInputDefs().size() == 0 &&
         node.OutputDefs().size() == 1;
}

const ONNX_NAMESPACE::AttributeProto* GetNodeAttribute(const Node& node, const std::string& attr_name) {
  const auto& attrs = node.GetAttributes();
  const auto iter = attrs.find(attr_name);
  return iter == attrs.end() ? nullptr : &iter->second;
}

bool RemoveNode(Graph& graph, Node& node) {
  // Cannot remove a node with implicit inputs, with multiple output NodeArgs (multiple output edges is fine),
  // or whose output is also a graph output.
  if (node.ImplicitInputDefs().size() > 0 ||
      node.OutputDefs().size() != 1 ||
      graph.IsNodeOutputsInGraphOutputs(node)) {
    return false;
  }

  if (node.GetInputEdgesCount() == 1) {
    // If there is a single input edge from another node (initializers are not connected with edges to nodes).
    return RemoveNodeWithSingleNodeIn(graph, node);
  } else if (node.InputDefs().size() == 1) {
    // If a single initializer is the only input.
    return RemoveNodeWithSingleInitializerIn(graph, node);
  } else {
    // No other node removal is supported, because there will be no way to connect its inputs to its outputs.
    return false;
  }
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
    // because it can be overriden by the user at runtime. For constant folding to be applied, the initializer should
    // not appear in the graph's inputs (that is the only way to guarantee it will always be constant).
    if (!graph.GetInitializedTensor(input_def->Name(), initializer) || IsGraphInput(graph, input_def)) {
      return false;
    }
  }
  return true;
}

size_t RemoveNodeOutputEdges(Graph& graph, Node& node) {
  std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);
  RemoveGraphEdges(graph, output_edges);

  return output_edges.size();
}

}  // namespace graph_utils

}  // namespace onnxruntime
